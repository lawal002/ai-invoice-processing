from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_pages"


class PreprocessingError(RuntimeError):
    pass


def preprocess_image(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Create a cleaned grayscale image for OCR."""
    input_path = Path(input_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path else PROCESSED_DIR / f"{input_path.stem}_preprocessed.png"

    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(input_path))
        if image is None:
            raise PreprocessingError(f"Could not read image: {input_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresholded = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        cv2.imwrite(str(output_path), thresholded)
        return output_path
    except ImportError:
        pass

    try:
        image = Image.open(input_path).convert("L")
        image = ImageOps.autocontrast(image)
        image = image.resize((int(image.width * 1.25), int(image.height * 1.25)))
        image.save(output_path)
        return output_path
    except OSError as exc:
        raise PreprocessingError(f"Could not preprocess image: {input_path}") from exc


def _deskew_gray(gray):
    """Rotate a grayscale image to correct small tilts detected via minAreaRect.

    Skips rotation when the detected angle is tiny (< 0.5°) or implausibly large
    (> 20°) to avoid mis-rotating upright documents or near-90° scans.
    Returns the original array unchanged when no rotation is applied.
    """
    import cv2
    import numpy as np

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = (90 + angle) if angle < -45 else angle
    if abs(angle) < 0.5 or abs(angle) > 20:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image_variants(input_path: str | Path) -> list[tuple[str, Path]]:
    """Create OCR preprocessing variants for low-confidence retry passes."""
    input_path = Path(input_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, Path]] = [("raw", input_path)]

    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(input_path))
        if image is None:
            raise PreprocessingError(f"Could not read image: {input_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remove saturated backgrounds without changing the canvas size.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        background_mask = (saturation > 70) & (value > 115)
        background_cleaned = image.copy()
        background_cleaned[background_mask] = [255, 255, 255]
        background_gray = cv2.cvtColor(background_cleaned, cv2.COLOR_BGR2GRAY)
        background_path = PROCESSED_DIR / f"{input_path.stem}_background_cleaned.png"
        cv2.imwrite(str(background_path), background_gray)
        variants.append(("background_cleaned", background_path))

        gray_path = PROCESSED_DIR / f"{input_path.stem}_gray.png"
        cv2.imwrite(str(gray_path), gray)
        variants.append(("grayscale", gray_path))

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        denoised_path = PROCESSED_DIR / f"{input_path.stem}_denoised.png"
        cv2.imwrite(str(denoised_path), denoised)
        variants.append(("denoised", denoised_path))

        contrast = cv2.equalizeHist(gray)
        contrast_path = PROCESSED_DIR / f"{input_path.stem}_contrast.png"
        cv2.imwrite(str(contrast_path), contrast)
        variants.append(("contrast", contrast_path))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        clahe_path = PROCESSED_DIR / f"{input_path.stem}_clahe.png"
        cv2.imwrite(str(clahe_path), clahe)
        variants.append(("clahe", clahe_path))

        _, otsu = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        otsu_path = PROCESSED_DIR / f"{input_path.stem}_otsu_threshold.png"
        cv2.imwrite(str(otsu_path), otsu)
        variants.append(("otsu_threshold", otsu_path))

        thresholded = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        threshold_path = PROCESSED_DIR / f"{input_path.stem}_adaptive_threshold.png"
        cv2.imwrite(str(threshold_path), thresholded)
        variants.append(("adaptive_threshold", threshold_path))

        resized = cv2.resize(contrast, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        resized_path = PROCESSED_DIR / f"{input_path.stem}_2x.png"
        cv2.imwrite(str(resized_path), resized)
        variants.append(("resized_2x", resized_path))

        clahe_resized = cv2.resize(clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe_resized_path = PROCESSED_DIR / f"{input_path.stem}_clahe_2x.png"
        cv2.imwrite(str(clahe_resized_path), clahe_resized)
        variants.append(("clahe_2x", clahe_resized_path))

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(resized, -1, kernel)
        sharpened_path = PROCESSED_DIR / f"{input_path.stem}_sharpened.png"
        cv2.imwrite(str(sharpened_path), sharpened)
        variants.append(("sharpened", sharpened_path))

        sharp_gray_2x = cv2.filter2D(
            cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
            -1,
            kernel,
        )
        sharp_gray_path = PROCESSED_DIR / f"{input_path.stem}_gray_sharpened_2x.png"
        cv2.imwrite(str(sharp_gray_path), sharp_gray_2x)
        variants.append(("gray_sharpened_2x", sharp_gray_path))

        # Correct small rotations only.
        deskewed = _deskew_gray(gray)
        if deskewed is not gray:
            deskew_path = PROCESSED_DIR / f"{input_path.stem}_deskewed.png"
            cv2.imwrite(str(deskew_path), deskewed)
            variants.append(("deskewed", deskew_path))

        # Plain upscale works better for some thermal and dot-matrix receipts.
        raw_up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        raw_up_path = PROCESSED_DIR / f"{input_path.stem}_raw_upscale.png"
        cv2.imwrite(str(raw_up_path), raw_up)
        variants.append(("raw_upscale", raw_up_path))

        return variants
    except ImportError:
        pass

    try:
        image = Image.open(input_path).convert("L")

        gray_path = PROCESSED_DIR / f"{input_path.stem}_gray.png"
        image.save(gray_path)
        variants.append(("grayscale", gray_path))

        denoised = image.filter(ImageFilter.MedianFilter(size=3))
        denoised_path = PROCESSED_DIR / f"{input_path.stem}_denoised.png"
        denoised.save(denoised_path)
        variants.append(("denoised", denoised_path))

        contrast = ImageOps.autocontrast(ImageEnhance.Contrast(image).enhance(1.8))
        contrast_path = PROCESSED_DIR / f"{input_path.stem}_contrast.png"
        contrast.save(contrast_path)
        variants.append(("contrast", contrast_path))

        sharp_contrast = ImageOps.autocontrast(ImageEnhance.Sharpness(contrast).enhance(2.0))
        sharp_contrast_path = PROCESSED_DIR / f"{input_path.stem}_clahe.png"
        sharp_contrast.save(sharp_contrast_path)
        variants.append(("clahe", sharp_contrast_path))

        otsu_like = ImageOps.autocontrast(image).point(lambda pixel: 255 if pixel > 140 else 0)
        otsu_path = PROCESSED_DIR / f"{input_path.stem}_otsu_threshold.png"
        otsu_like.save(otsu_path)
        variants.append(("otsu_threshold", otsu_path))

        thresholded = contrast.point(lambda pixel: 255 if pixel > 165 else 0)
        threshold_path = PROCESSED_DIR / f"{input_path.stem}_adaptive_threshold.png"
        thresholded.save(threshold_path)
        variants.append(("adaptive_threshold", threshold_path))

        resized = contrast.resize((image.width * 2, image.height * 2))
        resized_path = PROCESSED_DIR / f"{input_path.stem}_2x.png"
        resized.save(resized_path)
        variants.append(("resized_2x", resized_path))

        clahe_resized = sharp_contrast.resize((image.width * 2, image.height * 2))
        clahe_resized_path = PROCESSED_DIR / f"{input_path.stem}_clahe_2x.png"
        clahe_resized.save(clahe_resized_path)
        variants.append(("clahe_2x", clahe_resized_path))

        sharpened = resized.filter(ImageFilter.SHARPEN)
        sharpened_path = PROCESSED_DIR / f"{input_path.stem}_sharpened.png"
        sharpened.save(sharpened_path)
        variants.append(("sharpened", sharpened_path))

        gray_sharpened = image.resize((image.width * 2, image.height * 2)).filter(ImageFilter.SHARPEN)
        gray_sharpened_path = PROCESSED_DIR / f"{input_path.stem}_gray_sharpened_2x.png"
        gray_sharpened.save(gray_sharpened_path)
        variants.append(("gray_sharpened_2x", gray_sharpened_path))
        return variants
    except OSError as exc:
        raise PreprocessingError(f"Could not preprocess image variants: {input_path}") from exc
