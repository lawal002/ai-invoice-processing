from __future__ import annotations

import logging
import os
import re
import sys
import threading
import json
import time
import hashlib
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .preprocessing import PreprocessingError, preprocess_image, preprocess_image_variants

_logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WINDOWS_DLL_HANDLES: list[object] = []
_PADDLEOCR_IMPORT_LOCK = threading.Lock()
OCR_CACHE_VERSION = "ocr_cache_v1"


@dataclass
class OCRTokenData:
    text: str
    bbox: list[float]
    confidence: float
    page: int = 1
    source_variant: str = ""
    source_engine: str = ""


class OCRDependencyError(RuntimeError):
    pass


def _image_paths(path: Path) -> list[Path]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise OCRDependencyError("pdf2image is required for PDF OCR. Install dependencies first.") from exc
        out_dir = path.parent / f"{path.stem}_pages"
        out_dir.mkdir(exist_ok=True)
        try:
            pages = convert_from_path(str(path))
        except Exception as exc:
            raise OCRDependencyError(
                "PDF conversion failed. Install Poppler and ensure it is available on PATH, or upload JPG/PNG files."
            ) from exc
        image_paths = []
        for idx, page in enumerate(pages, start=1):
            image_path = out_dir / f"page_{idx:03d}.png"
            page.save(image_path)
            image_paths.append(image_path)
        return image_paths
    return [path]


def _preprocessed_paths(path: Path) -> list[Path]:
    processed_paths = []
    for page_path in _image_paths(path):
        try:
            processed_paths.append(preprocess_image(page_path))
        except PreprocessingError as exc:
            raise OCRDependencyError(str(exc)) from exc
    return processed_paths


def _ocr_max_dim() -> int:
    """Long-edge pixel cap before OCR.  Set AI_INVOICE_OCR_MAX_DIM to override (default 2000)."""
    return int(os.getenv("AI_INVOICE_OCR_MAX_DIM", "2000"))


def _ocr_timeout() -> int:
    """Per-page OCR timeout in seconds.  Set AI_INVOICE_OCR_TIMEOUT to override (default 120)."""
    return int(os.getenv("AI_INVOICE_OCR_TIMEOUT", "120"))


def _cap_resolution(image_path: Path, max_dim: int) -> Path:
    """Downscale image if its long edge exceeds max_dim, returning the (possibly new) path."""
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        return image_path  # graceful degradation if cv2 not available
    img = cv2.imread(str(image_path))
    if img is None:
        return image_path
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_dim:
        return image_path
    scale = max_dim / long_edge
    resized = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    out = image_path.parent / f"{image_path.stem}_capped.png"
    cv2.imwrite(str(out), resized)
    return out


def _run_with_timeout(fn, timeout_s: int, fallback):
    """Run fn() in a daemon thread; return fallback if it does not finish in timeout_s seconds."""
    result_holder: list = [fallback]
    exc_holder: list = [None]

    def _target() -> None:
        try:
            result_holder[0] = fn()
        except Exception as exc:  # noqa: BLE001
            exc_holder[0] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        _logger.warning(
            "OCR inference timed out after %ds — returning partial/empty result", timeout_s
        )
        return fallback
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


def _low_confidence_threshold() -> float:
    return float(os.getenv("AI_INVOICE_LOW_CONFIDENCE_THRESHOLD", "0.55"))


def _average_confidence(tokens: list[OCRTokenData]) -> float:
    if not tokens:
        return 0.0
    return mean(max(min(token.confidence, 1.0), 0.0) for token in tokens)


def _paddle_preprocessing_enabled() -> bool:
    return os.getenv("AI_INVOICE_PADDLE_PREPROCESSING", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _paddle_fusion_enabled() -> bool:
    return os.getenv("AI_INVOICE_PADDLE_FUSION", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _paddle_force_fusion_enabled() -> bool:
    return os.getenv("AI_INVOICE_PADDLE_FORCE_FUSION", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _ocr_cache_enabled() -> bool:
    return _env_flag("AI_INVOICE_OCR_CACHE", True)


def _ocr_cache_dir() -> Path:
    configured = os.getenv("AI_INVOICE_OCR_CACHE_DIR", "").strip()
    return Path(configured) if configured else PROJECT_ROOT / "data" / "processed" / "ocr_cache"


def _paddle_use_doc_orientation() -> bool:
    return _env_flag("AI_INVOICE_PADDLE_DOC_ORIENTATION", False)


def _paddle_use_doc_unwarping() -> bool:
    return _env_flag("AI_INVOICE_PADDLE_DOC_UNWARPING", False)


def _paddle_use_textline_orientation() -> bool:
    return _env_flag("AI_INVOICE_PADDLE_TEXTLINE_ORIENTATION", False)


def _paddleocr_v3_kwargs(lang: str) -> dict:
    return {
        "lang": lang,
        "use_doc_orientation_classify": _paddle_use_doc_orientation(),
        "use_doc_unwarping": _paddle_use_doc_unwarping(),
        "use_textline_orientation": _paddle_use_textline_orientation(),
    }


def _ocr_debug_enabled() -> bool:
    return os.getenv("AI_INVOICE_OCR_DEBUG", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _ocr_debug_dir() -> Path:
    configured = os.getenv("AI_INVOICE_OCR_DEBUG_DIR", "").strip()
    return Path(configured) if configured else PROJECT_ROOT / "results" / "ocr_debug"


def _paddle_cjk_single_pass_enabled() -> bool:
    """Allow users to trade Chinese OCR quality for speed when needed.

    Default is multi-pass for CJK too because real VAT/fapiao photos often need
    preprocessing variants to recover faded Chinese/numeric fields.
    """
    return os.getenv("AI_INVOICE_PADDLE_CJK_SINGLE_PASS", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _ocr_variant_limit() -> int:
    value = os.getenv("AI_INVOICE_OCR_VARIANT_LIMIT", "").strip()
    if not value:
        return 0
    try:
        return max(int(value), 1)
    except ValueError:
        return 0


def _has_chinese_language(languages: tuple[str, ...]) -> bool:
    return any(lang.lower().replace("-", "_") in {"ch", "ch_sim", "zh", "zh_cn", "zh_sim"} for lang in languages)


def _auto_chinese_ocr_enabled() -> bool:
    return os.getenv("AI_INVOICE_AUTO_CHINESE_OCR", "true").strip().lower() not in {"0", "false", "no", "off"}


def _chinese_retry_languages(languages: tuple[str, ...]) -> tuple[str, ...]:
    ordered = ["ch_sim", "en"]
    for lang in languages:
        if lang not in ordered:
            ordered.append(lang)
    return tuple(ordered)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _tokens_look_like_chinese_invoice(tokens: list[OCRTokenData]) -> bool:
    if not tokens:
        return False
    min_y = min(token.bbox[1] for token in tokens)
    max_y = max(token.bbox[3] for token in tokens)
    height = max(max_y - min_y, 1.0)

    has_cjk = any(_contains_cjk(token.text) for token in tokens)
    has_header_code = False
    has_date = False
    has_amount = False
    has_taxi_plate = False
    clear_latin_labels = 0

    for token in tokens:
        text = token.text or ""
        y_ratio = ((token.bbox[1] + token.bbox[3]) / 2 - min_y) / height
        if y_ratio <= 0.45 and any(len(run) >= 10 for run in re.findall(r"\d{10,14}", text)):
            has_header_code = True
        if re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", text):
            has_date = True
        if re.search(r"\b\d+[,.]\d{2}\b", text):
            has_amount = True
        if re.search(r"\b[A-Z]{1,3}-\d{3,5}\b", text, re.I):
            has_taxi_plate = True
        if re.search(r"\b(invoice|receipt|total|subtotal|tax|vendor|date)\b", text, re.I):
            clear_latin_labels += 1

    low_quality = _average_confidence(tokens) < 0.58
    noisy_fapiao_profile = has_header_code and has_date and has_amount and (has_taxi_plate or low_quality)
    return has_cjk or (noisy_fapiao_profile and clear_latin_labels <= 2)


SUPPORTED_ENGINES = ("auto", "paddleocr", "easyocr", "tesseract")


def _default_engine() -> str:
    """Return the OCR engine configured for the web pipeline.

    `auto` is the default: try PaddleOCR first, then fall back to EasyOCR when
    PaddleOCR is unavailable. Use AI_INVOICE_OCR_ENGINE=paddleocr to force
    PaddleOCR and surface dependency/model errors directly.
    """
    engine = os.getenv("AI_INVOICE_OCR_ENGINE", "auto").strip().lower()
    return engine or "auto"


def _default_languages(engine: str = "easyocr") -> tuple[str, ...]:
    """Return the per-document language list from the environment.

    Set AI_INVOICE_OCR_LANGS as a comma-separated list of EasyOCR/PaddleOCR
    language codes (e.g. "ch_sim,en" for Chinese+English).

    PaddleOCR's Chinese model handles Chinese plus Latin digits/letters, so it
    is the default for PaddleOCR. EasyOCR stays English by default because its
    Chinese model is a separate download and slower to load.
    """
    default = "ch_sim,en" if engine == "paddleocr" else "en"
    raw = os.getenv("AI_INVOICE_OCR_LANGS", default).strip()
    langs = tuple(x.strip() for x in raw.split(",") if x.strip())
    return langs or tuple(default.split(","))


def _path_language_hint(path: Path, engine: str) -> tuple[tuple[str, ...], str] | tuple[None, None]:
    """Infer a cheap OCR language hint from the dataset/file path.

    This is intentionally lightweight: it only affects unlabeled local paths
    with obvious category names and never overrides AI_INVOICE_OCR_LANGS or an
    explicit languages= argument.
    """
    joined = " ".join(part.lower() for part in (*path.parts[-5:], path.stem.lower()))
    chinese_markers = (
        "chinese",
        "vatid",
        "vat",
        "fapiao",
        "taxi",
        "cny",
    )
    latin_markers = (
        "malaysian",
        "english",
        "fatura",
        "clean",
        "noisy",
        "receipt",
        "invoice",
    )
    if any(marker in joined for marker in chinese_markers):
        return ("ch_sim", "en"), "path_category_hint:chinese"
    if any(marker in joined for marker in latin_markers):
        return ("en",), "path_category_hint:latin"
    return None, None


def _resolve_ocr_languages(
    path: Path,
    engine: str,
    languages: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], str]:
    if languages:
        return tuple(languages), "explicit"
    if os.getenv("AI_INVOICE_OCR_LANGS", "").strip():
        return _default_languages(engine), "environment"
    hinted_languages, hint_source = _path_language_hint(path, engine)
    if hinted_languages:
        return hinted_languages, hint_source or "path_category_hint"
    return _default_languages(engine), "engine_default"


def _engine_sequence(engine: str) -> tuple[str, ...]:
    engine = (engine or "auto").strip().lower()
    if engine == "auto":
        return ("paddleocr", "easyocr")
    if engine in SUPPORTED_ENGINES and engine != "auto":
        return (engine,)
    raise OCRDependencyError(f"Unsupported OCR engine: {engine!r}. Supported: {SUPPORTED_ENGINES}")


def _add_windows_nvidia_dll_dirs() -> None:
    """Expose venv-installed NVIDIA runtime DLLs to Paddle/PaddleOCR on Windows."""
    if os.name != "nt":
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        return
    dll_dirs = [
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
        nvidia_root / "cudnn" / "bin",
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cufft" / "bin",
        nvidia_root / "curand" / "bin",
        nvidia_root / "cusolver" / "bin",
        nvidia_root / "cusparse" / "bin",
        nvidia_root / "nvjitlink" / "bin",
    ]
    existing_path = os.environ.get("PATH", "")
    path_parts = existing_path.split(os.pathsep) if existing_path else []
    changed = False
    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue
        dll_dir_str = str(dll_dir)
        if dll_dir_str not in path_parts:
            path_parts.insert(0, dll_dir_str)
            changed = True
        try:
            handle = os.add_dll_directory(dll_dir_str)
        except (AttributeError, FileNotFoundError, OSError):
            continue
        _WINDOWS_DLL_HANDLES.append(handle)
    if changed:
        os.environ["PATH"] = os.pathsep.join(path_parts)


def _alternate_engine_retry_enabled() -> bool:
    return os.getenv("AI_INVOICE_ALT_ENGINE_RETRY", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _alternate_engine_retry_confidence() -> float:
    return float(os.getenv("AI_INVOICE_ALT_ENGINE_RETRY_CONFIDENCE", "0.45"))


def _alternate_engine_retry_min_tokens() -> int:
    return int(os.getenv("AI_INVOICE_ALT_ENGINE_RETRY_MIN_TOKENS", "5"))


def _should_retry_with_alternate_engine(tokens: list[OCRTokenData]) -> bool:
    if not _alternate_engine_retry_enabled():
        return False
    metrics = _ocr_quality_metrics(tokens)
    token_count = int(metrics["token_count"])
    average_confidence = float(metrics["average_confidence"])
    if token_count == 0:
        return True
    if average_confidence < _alternate_engine_retry_confidence():
        return True
    if token_count < _alternate_engine_retry_min_tokens() and average_confidence < 0.75:
        return True
    if float(metrics["quality"]) < 0.50:
        return True
    if token_count >= 8 and int(metrics["garbage_count"]) >= max(4, token_count // 3):
        return True
    return False


def _prefer_alternate_engine_tokens(primary: list[OCRTokenData], alternate: list[OCRTokenData]) -> bool:
    if not alternate:
        return False
    if not primary:
        return True
    primary_metrics = _ocr_quality_metrics(primary)
    alternate_metrics = _ocr_quality_metrics(alternate)
    primary_quality = float(primary_metrics["quality"])
    alternate_quality = float(alternate_metrics["quality"])
    primary_conf = float(primary_metrics["average_confidence"])
    alternate_conf = float(alternate_metrics["average_confidence"])
    primary_count = int(primary_metrics["token_count"])
    alternate_count = int(alternate_metrics["token_count"])

    if alternate_quality >= primary_quality * 1.15 and alternate_conf >= primary_conf * 0.90:
        return True
    if primary_count < 5 and alternate_count >= primary_count + 5 and alternate_conf >= 0.35:
        return True
    if primary_conf < 0.30 and alternate_conf >= 0.45 and alternate_count >= primary_count:
        return True
    return False


def _annotate_tokens(tokens: list[OCRTokenData], engine: str) -> list[OCRTokenData]:
    for token in tokens:
        if not getattr(token, "source_engine", ""):
            token.source_engine = engine
    return tokens


def _counts(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: item[0]))


def _expanded_variant_names(tokens: list[OCRTokenData]) -> list[str]:
    variants: list[str] = []
    for token in tokens:
        for part in (getattr(token, "source_variant", "") or "").split("+"):
            part = part.strip()
            if part:
                variants.append(part)
    return variants


def _token_to_cache_dict(token: OCRTokenData) -> dict:
    return {
        "text": token.text,
        "bbox": [float(value) for value in token.bbox[:4]],
        "confidence": float(token.confidence),
        "page": int(token.page),
        "source_variant": token.source_variant,
        "source_engine": token.source_engine,
    }


def _token_from_cache_dict(value: dict) -> OCRTokenData:
    bbox = value.get("bbox") or [0, 0, 0, 0]
    return OCRTokenData(
        text=str(value.get("text") or ""),
        bbox=[float(item or 0.0) for item in bbox[:4]],
        confidence=float(value.get("confidence") or 0.0),
        page=int(value.get("page") or 1),
        source_variant=str(value.get("source_variant") or ""),
        source_engine=str(value.get("source_engine") or ""),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ocr_cache_env_settings() -> dict[str, str]:
    names = (
        "AI_INVOICE_OCR_LANGS",
        "AI_INVOICE_OCR_MAX_DIM",
        "AI_INVOICE_LOW_CONFIDENCE_THRESHOLD",
        "AI_INVOICE_ALT_ENGINE_RETRY_CONFIDENCE",
        "AI_INVOICE_ALT_ENGINE_RETRY_MIN_TOKENS",
        "AI_INVOICE_AUTO_CHINESE_OCR",
        "AI_INVOICE_PADDLE_PREPROCESSING",
        "AI_INVOICE_PADDLE_FUSION",
        "AI_INVOICE_PADDLE_FORCE_FUSION",
        "AI_INVOICE_PADDLE_CJK_SINGLE_PASS",
        "AI_INVOICE_OCR_VARIANT_LIMIT",
        "AI_INVOICE_PADDLE_DOC_ORIENTATION",
        "AI_INVOICE_PADDLE_DOC_UNWARPING",
        "AI_INVOICE_PADDLE_TEXTLINE_ORIENTATION",
        "AI_INVOICE_OCR_GPU",
    )
    return {name: os.getenv(name, "") for name in names}


def _ocr_cache_key(path: Path, requested_engine: str, languages: tuple[str, ...] | None) -> str:
    if not path.exists() or not path.is_file():
        return ""
    stat = path.stat()
    payload = {
        "version": OCR_CACHE_VERSION,
        "file_sha256": _file_sha256(path),
        "file_size": stat.st_size,
        "file_suffix": path.suffix.lower(),
        "path_hint": [part.lower() for part in path.parts[-5:]],
        "requested_engine": requested_engine,
        "languages_arg": list(languages or ()),
        "env": _ocr_cache_env_settings(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ocr_cache_path(cache_key: str) -> Path:
    return _ocr_cache_dir() / f"{cache_key}.json"


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _metadata_with_cache_info(metadata: dict, *, cache_key: str, cache_path: Path, hit: bool) -> dict:
    updated = dict(metadata)
    updated["cache"] = {
        "enabled": True,
        "hit": hit,
        "cache_key": cache_key,
        "cache_path": _relative_to_project(cache_path),
        "cache_version": OCR_CACHE_VERSION,
        "loaded_at": datetime.now(timezone.utc).isoformat() if hit else "",
    }
    updated["loaded_from_cache"] = hit
    return updated


def _load_ocr_cache(cache_key: str) -> tuple[list[OCRTokenData], dict] | None:
    if not cache_key:
        return None
    cache_path = _ocr_cache_path(cache_key)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != OCR_CACHE_VERSION:
            return None
        tokens = [_token_from_cache_dict(item) for item in payload.get("tokens", [])]
        metadata = payload.get("metadata") or {}
        return tokens, _metadata_with_cache_info(metadata, cache_key=cache_key, cache_path=cache_path, hit=True)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _logger.warning("Ignoring unreadable OCR cache entry %s: %s", cache_path, exc)
        return None


def _store_ocr_cache(cache_key: str, tokens: list[OCRTokenData], metadata: dict) -> dict:
    if not cache_key:
        return metadata
    cache_path = _ocr_cache_path(cache_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_with_cache_info(metadata, cache_key=cache_key, cache_path=cache_path, hit=False)
    payload = {
        "version": OCR_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "tokens": [_token_to_cache_dict(token) for token in tokens],
    }
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(cache_path)
    except OSError as exc:
        _logger.warning("Could not write OCR cache entry %s: %s", cache_path, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return metadata


def ocr_metadata_from_tokens(
    tokens: list[OCRTokenData],
    *,
    requested_engine: str = "",
    selected_engine: str = "",
    languages: tuple[str, ...] = (),
    language_source: str = "",
    attempts: list[dict] | None = None,
    fallback_reason: str = "",
) -> dict:
    metrics = _ocr_quality_metrics(tokens)
    engines = _counts([getattr(token, "source_engine", "") for token in tokens])
    variants = _counts(_expanded_variant_names(tokens))
    if not selected_engine and engines:
        selected_engine = max(engines.items(), key=lambda item: item[1])[0]
    return {
        "requested_engine": requested_engine,
        "selected_engine": selected_engine,
        "languages": list(languages),
        "language_source": language_source,
        "metrics": metrics,
        "token_count": int(metrics["token_count"]),
        "average_confidence": float(metrics["average_confidence"]),
        "source_engines": engines,
        "source_variants": variants,
        "preprocessing_variants": sorted(variants),
        "fallback_reason": fallback_reason,
        "attempts": attempts or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_engine(path: Path, engine: str, languages: tuple[str, ...]) -> list[OCRTokenData]:
    if engine == "paddleocr":
        return _annotate_tokens(_run_paddleocr(path, languages), engine)
    if engine == "easyocr":
        return _annotate_tokens(_run_easyocr(path, languages), engine)
    if engine == "tesseract":
        return _annotate_tokens(_run_tesseract(path), engine)
    raise OCRDependencyError(f"Unsupported OCR engine: {engine!r}. Supported: {SUPPORTED_ENGINES}")


def _attempt_record(
    *,
    engine: str,
    languages: tuple[str, ...],
    language_source: str,
    elapsed_ms: float,
    tokens: list[OCRTokenData] | None = None,
    error: str = "",
    retried_with_alternate: bool = False,
) -> dict:
    record = {
        "engine": engine,
        "languages": list(languages),
        "language_source": language_source,
        "elapsed_ms": round(float(elapsed_ms), 3),
        "retried_with_alternate": retried_with_alternate,
    }
    if tokens is not None:
        record["metrics"] = _ocr_quality_metrics(tokens)
        record["source_variants"] = _counts(_expanded_variant_names(tokens))
    if error:
        record["error"] = error
    return record


def run_ocr_with_metadata(
    path: str | Path,
    languages: tuple[str, ...] | None = None,
    engine: str | None = None,
) -> tuple[list[OCRTokenData], dict]:
    requested_engine = (engine or _default_engine()).strip().lower()
    path = Path(path)
    cache_key = _ocr_cache_key(path, requested_engine, languages) if _ocr_cache_enabled() else ""
    cached = _load_ocr_cache(cache_key) if cache_key else None
    if cached is not None:
        return cached

    errors: list[str] = []
    attempts: list[dict] = []

    if requested_engine != "auto":
        selected_languages, language_source = _resolve_ocr_languages(path, requested_engine, languages)
        started = time.perf_counter()
        tokens = _run_engine(path, requested_engine, selected_languages)
        elapsed_ms = (time.perf_counter() - started) * 1000
        attempts.append(
            _attempt_record(
                engine=requested_engine,
                languages=selected_languages,
                language_source=language_source,
                elapsed_ms=elapsed_ms,
                tokens=tokens,
            )
        )
        metadata = ocr_metadata_from_tokens(
            tokens,
            requested_engine=requested_engine,
            selected_engine=requested_engine,
            languages=selected_languages,
            language_source=language_source,
            attempts=attempts,
        )
        if cache_key:
            metadata = _store_ocr_cache(cache_key, tokens, metadata)
        return tokens, metadata

    paddle_tokens: list[OCRTokenData] | None = None
    paddle_languages: tuple[str, ...] = ()
    paddle_language_source = ""
    fallback_reason = ""

    paddle_languages, paddle_language_source = _resolve_ocr_languages(path, "paddleocr", languages)
    try:
        started = time.perf_counter()
        paddle_tokens = _run_engine(path, "paddleocr", paddle_languages)
        elapsed_ms = (time.perf_counter() - started) * 1000
        should_retry = _should_retry_with_alternate_engine(paddle_tokens)
        if should_retry:
            fallback_reason = "paddle_low_quality"
        attempts.append(
            _attempt_record(
                engine="paddleocr",
                languages=paddle_languages,
                language_source=paddle_language_source,
                elapsed_ms=elapsed_ms,
                tokens=paddle_tokens,
                retried_with_alternate=should_retry,
            )
        )
        if not should_retry:
            metadata = ocr_metadata_from_tokens(
                paddle_tokens,
                requested_engine=requested_engine,
                selected_engine="paddleocr",
                languages=paddle_languages,
                language_source=paddle_language_source,
                attempts=attempts,
            )
            if cache_key:
                metadata = _store_ocr_cache(cache_key, paddle_tokens, metadata)
            return paddle_tokens, metadata
    except OCRDependencyError as exc:
        errors.append(f"paddleocr: {exc}")
        attempts.append(
            _attempt_record(
                engine="paddleocr",
                languages=paddle_languages,
                language_source=paddle_language_source,
                elapsed_ms=0.0,
                error=str(exc),
                retried_with_alternate=True,
            )
        )
        fallback_reason = "paddle_error"

    easy_languages, easy_language_source = _resolve_ocr_languages(path, "easyocr", languages)
    if (
        paddle_tokens
        and _tokens_look_like_chinese_invoice(paddle_tokens)
        and not _has_chinese_language(easy_languages)
    ):
        easy_languages = _chinese_retry_languages(easy_languages)
        easy_language_source = "paddle_chinese_profile_retry"
    try:
        started = time.perf_counter()
        easy_tokens = _run_engine(path, "easyocr", easy_languages)
        elapsed_ms = (time.perf_counter() - started) * 1000
        attempts.append(
            _attempt_record(
                engine="easyocr",
                languages=easy_languages,
                language_source=easy_language_source,
                elapsed_ms=elapsed_ms,
                tokens=easy_tokens,
            )
        )
    except OCRDependencyError as exc:
        errors.append(f"easyocr: {exc}")
        attempts.append(
            _attempt_record(
                engine="easyocr",
                languages=easy_languages,
                language_source=easy_language_source,
                elapsed_ms=0.0,
                error=str(exc),
            )
        )
        if paddle_tokens is not None:
            metadata = ocr_metadata_from_tokens(
                paddle_tokens,
                requested_engine=requested_engine,
                selected_engine="paddleocr",
                languages=paddle_languages,
                language_source=paddle_language_source,
                attempts=attempts,
                fallback_reason=fallback_reason + "_fallback_failed",
            )
            if cache_key:
                metadata = _store_ocr_cache(cache_key, paddle_tokens, metadata)
            return paddle_tokens, metadata
        raise OCRDependencyError("All OCR engines failed. " + " | ".join(errors)) from exc

    if paddle_tokens is None or _prefer_alternate_engine_tokens(paddle_tokens, easy_tokens):
        selected_tokens = easy_tokens
        selected_engine = "easyocr"
        selected_languages = easy_languages
        selected_language_source = easy_language_source
    else:
        selected_tokens = paddle_tokens
        selected_engine = "paddleocr"
        selected_languages = paddle_languages
        selected_language_source = paddle_language_source

    metadata = ocr_metadata_from_tokens(
        selected_tokens,
        requested_engine=requested_engine,
        selected_engine=selected_engine,
        languages=selected_languages,
        language_source=selected_language_source,
        attempts=attempts,
        fallback_reason=fallback_reason,
    )
    if cache_key:
        metadata = _store_ocr_cache(cache_key, selected_tokens, metadata)
    return selected_tokens, metadata


def run_ocr(path: str | Path, languages: tuple[str, ...] | None = None, engine: str | None = None) -> list[OCRTokenData]:
    tokens, _ = run_ocr_with_metadata(path, languages=languages, engine=engine)
    return tokens


def _easyocr_should_use_gpu() -> bool:
    """Use CUDA automatically when available, with an environment override.

    Set AI_INVOICE_OCR_GPU=false to force CPU, or true to request GPU.
    The default is auto-detect.
    """
    override = os.getenv("AI_INVOICE_OCR_GPU", "auto").strip().lower()
    if override in {"0", "false", "no", "cpu"}:
        return False
    if override in {"1", "true", "yes", "gpu", "cuda"}:
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


@lru_cache(maxsize=4)
def _get_easyocr_reader(languages: tuple[str, ...], use_gpu: bool):
    import easyocr

    return easyocr.Reader(list(languages), gpu=use_gpu)


def _read_easyocr_image(reader, image_path: Path, page_number: int, timeout_s: int = 120) -> list[OCRTokenData]:
    def _infer():
        return reader.readtext(str(image_path))

    raw_results = _run_with_timeout(_infer, timeout_s, [])
    tokens: list[OCRTokenData] = []
    for bbox, text, confidence in raw_results:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        tokens.append(
            OCRTokenData(
                text=text,
                bbox=[float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                confidence=float(confidence),
                page=page_number,
                source_engine="easyocr",
            )
        )
    return tokens


def _ocr_quality(tokens: list[OCRTokenData]) -> float:
    metrics = _ocr_quality_metrics(tokens)
    return float(metrics["quality"])


def _token_signal_score(token: OCRTokenData) -> float:
    text = token.text or ""
    compact = re.sub(r"\s+", "", text)
    score = 0.0
    if re.search(r"\d+[,.]\d{2}", text):
        score += 1.20
    if re.search(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", text) or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text):
        score += 1.00
    if re.search(r"\d{6,}", text):
        score += 0.80
    if re.search(r"\b(?:RM|MYR|USD|EUR|CNY|RMB|NGN)\b|[$¥€₦]|元", text, re.I):
        score += 0.70
    if re.search(r"\b(?:invoice|receipt|total|subtotal|tax|vat|gst|date|amount)\b", text, re.I):
        score += 0.45
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    if cjk_count:
        score += min(cjk_count, 12) / 6.0
    if len(compact) >= 4 and not re.search(r"[A-Za-z0-9\u3400-\u9fff]", compact):
        score -= 0.50
    return score


def _ocr_quality_metrics(tokens: list[OCRTokenData]) -> dict[str, float | int]:
    if not tokens:
        return {
            "quality": 0.0,
            "average_confidence": 0.0,
            "token_count": 0,
            "high_confidence_count": 0,
            "low_confidence_count": 0,
            "signal_score": 0.0,
            "cjk_token_count": 0,
            "garbage_count": 0,
        }
    average_confidence = _average_confidence(tokens)
    high_conf_count = sum(1 for token in tokens if token.confidence >= 0.30)
    low_conf_count = sum(1 for token in tokens if token.confidence < 0.15)
    signal_score = sum(_token_signal_score(token) * max(min(token.confidence, 1.0), 0.05) for token in tokens)
    cjk_token_count = sum(1 for token in tokens if _contains_cjk(token.text))
    garbage_count = sum(
        1
        for token in tokens
        if token.confidence < 0.20
        and len((token.text or "").strip()) >= 4
        and _token_signal_score(token) <= 0
    )
    quality = (
        average_confidence
        + min(high_conf_count, 80) / 360.0
        + min(signal_score, 80.0) / 48.0
        + min(cjk_token_count, 30) / 90.0
        - min(low_conf_count, 120) / 500.0
        - min(garbage_count, 80) / 260.0
    )
    return {
        "quality": round(float(quality), 6),
        "average_confidence": round(float(average_confidence), 6),
        "token_count": len(tokens),
        "high_confidence_count": high_conf_count,
        "low_confidence_count": low_conf_count,
        "signal_score": round(float(signal_score), 6),
        "cjk_token_count": cjk_token_count,
        "garbage_count": garbage_count,
    }


def _chinese_ocr_quality(tokens: list[OCRTokenData]) -> float:
    cjk_count = sum(1 for token in tokens if _contains_cjk(token.text))
    chinese_label_bonus = sum(
        1
        for token in tokens
        if any(fragment in (token.text or "") for fragment in ("发票", "金额", "日期", "税", "价税", "号码"))
    )
    return _ocr_quality(tokens) + min(cjk_count, 40) / 80.0 + min(chinese_label_bonus, 12) / 18.0


def _best_easyocr_tokens_for_variants(
    reader,
    variants: list[tuple[str, Path]],
    page_number: int,
    timeout_s: int = 120,
) -> list[OCRTokenData]:
    raw_name, raw_path = variants[0]
    best_tokens = [
        OCRTokenData(token.text, list(token.bbox), token.confidence, token.page, raw_name, token.source_engine or "easyocr")
        for token in _read_easyocr_image(reader, raw_path, page_number, timeout_s=timeout_s)
    ]
    variant_records: list[dict] = [
        {
            "name": raw_name,
            "path": str(raw_path),
            "metrics": _ocr_quality_metrics(best_tokens),
            "tokens": [_token_to_debug_dict(token) for token in best_tokens],
        }
    ]
    selection_mode = "raw_only"
    if _average_confidence(best_tokens) < _low_confidence_threshold() or len(best_tokens) < 5:
        for variant_name, variant_path in variants[1:]:
            variant_tokens = [
                OCRTokenData(token.text, list(token.bbox), token.confidence, token.page, variant_name, token.source_engine or "easyocr")
                for token in _read_easyocr_image(reader, variant_path, page_number, timeout_s=timeout_s)
            ]
            variant_records.append(
                {
                    "name": variant_name,
                    "path": str(variant_path),
                    "metrics": _ocr_quality_metrics(variant_tokens),
                    "tokens": [_token_to_debug_dict(token) for token in variant_tokens],
                }
            )
            if _ocr_quality(variant_tokens) > _ocr_quality(best_tokens):
                best_tokens = variant_tokens
                selection_mode = "best_single_variant"
    _write_ocr_debug_report(
        engine="easyocr",
        reference_path=raw_path,
        page_number=page_number,
        variant_records=variant_records,
        selected_tokens=best_tokens,
        selection_mode=selection_mode,
    )
    return best_tokens


def _should_retry_chinese_easyocr(tokens: list[OCRTokenData], languages: tuple[str, ...]) -> bool:
    return (
        _auto_chinese_ocr_enabled()
        and not _has_chinese_language(languages)
        and _tokens_look_like_chinese_invoice(tokens)
    )


def _prefer_chinese_easyocr_tokens(primary: list[OCRTokenData], chinese: list[OCRTokenData]) -> bool:
    if not chinese:
        return False
    if sum(1 for token in chinese if _contains_cjk(token.text)) >= 2:
        return _chinese_ocr_quality(chinese) >= _ocr_quality(primary) * 0.75
    return _ocr_quality(chinese) > _ocr_quality(primary) * 1.15


def _run_easyocr(path: Path, languages: tuple[str, ...]) -> list[OCRTokenData]:
    try:
        import easyocr
    except ImportError as exc:
        raise OCRDependencyError("EasyOCR is not installed. Run `pip install -r requirements.txt`.") from exc
    except OSError as exc:
        raise OCRDependencyError(f"EasyOCR/Torch could not load required runtime DLLs: {exc}") from exc

    use_gpu = _easyocr_should_use_gpu()
    try:
        reader = _get_easyocr_reader(tuple(languages), use_gpu)
    except OSError as exc:
        raise OCRDependencyError(f"EasyOCR/Torch could not load required runtime DLLs: {exc}") from exc
    timeout_s = _ocr_timeout()
    tokens: list[OCRTokenData] = []
    for page_number, page_path in enumerate(_image_paths(path), start=1):
        page_path = _cap_resolution(page_path, _ocr_max_dim())
        try:
            variants = preprocess_image_variants(page_path)
        except PreprocessingError as exc:
            raise OCRDependencyError(str(exc)) from exc

        best_tokens = _best_easyocr_tokens_for_variants(reader, variants, page_number, timeout_s=timeout_s)
        if _should_retry_chinese_easyocr(best_tokens, tuple(languages)):
            retry_reader = _get_easyocr_reader(_chinese_retry_languages(tuple(languages)), use_gpu)
            retry_tokens = _best_easyocr_tokens_for_variants(retry_reader, variants, page_number, timeout_s=timeout_s)
            if _prefer_chinese_easyocr_tokens(best_tokens, retry_tokens):
                best_tokens = retry_tokens
        tokens.extend(best_tokens)
    return tokens


# PaddleOCR language code map — covers both EasyOCR aliases and native paddle codes.
# "ch_sim" is EasyOCR's Chinese code; PaddleOCR uses "ch".
_PADDLE_LANG_MAP: dict[str, str] = {
    "en": "en", "ch": "ch", "ch_sim": "ch", "zh": "ch",
    "fr": "fr", "de": "german", "german": "german",
    "ja": "japan", "ko": "korean", "ar": "arabic",
}


@lru_cache(maxsize=2)
def _get_paddleocr_reader(lang: str):
    _add_windows_nvidia_dll_dirs()
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def find_spec_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            return None
        return original_find_spec(name, *args, **kwargs)

    try:
        with _PADDLEOCR_IMPORT_LOCK:
            # PaddleOCR 3.x imports ModelScope, whose logger probes Torch at import
            # time. In this project Torch and Paddle may use different CUDA DLL
            # stacks, so keep Torch out of the PaddleOCR import path.
            importlib.util.find_spec = find_spec_without_torch
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR  # type: ignore[import]
            import logging

            logging.getLogger("ppocr").setLevel(logging.ERROR)
            try:
                return PaddleOCR(**_paddleocr_v3_kwargs(lang))                    # 3.x
            except TypeError:
                return PaddleOCR(
                    use_angle_cls=_paddle_use_textline_orientation(),
                    lang=lang,
                    show_log=False,
                )                                                                # 2.x
    except ImportError as exc:
        raise OCRDependencyError(
            "PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle"
        ) from exc
    except OSError as exc:
        raise OCRDependencyError(
            "PaddleOCR/PaddlePaddle could not load the required CUDA DLLs. "
            f"Original error: {exc}"
        ) from exc
    finally:
        importlib.util.find_spec = original_find_spec


def _paddle_tokens_from_result(result, page_number: int) -> list[OCRTokenData]:
    out: list[OCRTokenData] = []
    if not result:
        return out
    # PaddleOCR 3.x result shape.
    if isinstance(result, list) and result and hasattr(result[0], "get"):
        for res in result:
            texts  = res.get("rec_texts")  or res.get("rec_text")  or []
            scores = res.get("rec_scores") or res.get("rec_score") or []
            polys  = res.get("rec_polys")  or res.get("dt_polys")  or []
            for text, score, poly in zip(texts, scores, polys):
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                t = (text or "").strip()
                if t:
                    out.append(OCRTokenData(t, [min(xs), min(ys), max(xs), max(ys)],
                                            float(score), page_number, source_engine="paddleocr"))
        return out
    # PaddleOCR 2.x result shape.
    page = result[0] if (isinstance(result[0], list)) else result
    for item in (page or []):
        if not item:
            continue
        box, (text, conf) = item
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        t = (text or "").strip()
        if t:
            out.append(OCRTokenData(t, [min(xs), min(ys), max(xs), max(ys)],
                                    float(conf), page_number, source_engine="paddleocr"))
    return out


def _read_paddle_image(reader, image_path: Path, page_number: int, timeout_s: int = 120) -> list[OCRTokenData]:
    def _infer():
        if hasattr(reader, "predict"):
            return reader.predict(str(image_path))   # 3.x
        return reader.ocr(str(image_path), cls=True)  # 2.x

    try:
        result = _run_with_timeout(_infer, timeout_s, None)
    except Exception as exc:
        raise OCRDependencyError(f"PaddleOCR inference failed: {exc}") from exc
    if result is None:
        return []
    return _paddle_tokens_from_result(result, page_number)


def _safe_image_size(path: Path) -> tuple[float, float] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return float(image.width), float(image.height)
    except Exception:
        return None


def _scale_tokens_to_reference(
    tokens: list[OCRTokenData],
    variant_name: str,
    variant_path: Path,
    reference_path: Path,
) -> list[OCRTokenData]:
    reference_size = _safe_image_size(reference_path)
    variant_size = _safe_image_size(variant_path)
    if not reference_size or not variant_size:
        return [
            OCRTokenData(t.text, list(t.bbox), t.confidence, t.page, variant_name, t.source_engine)
            for t in tokens
        ]
    ref_w, ref_h = reference_size
    var_w, var_h = variant_size
    if var_w <= 0 or var_h <= 0:
        return [
            OCRTokenData(t.text, list(t.bbox), t.confidence, t.page, variant_name, t.source_engine)
            for t in tokens
        ]
    sx = ref_w / var_w
    sy = ref_h / var_h
    scaled: list[OCRTokenData] = []
    for token in tokens:
        x1, y1, x2, y2 = token.bbox
        scaled.append(
            OCRTokenData(
                text=token.text,
                bbox=[x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                confidence=token.confidence,
                page=token.page,
                source_variant=variant_name,
                source_engine=token.source_engine,
            )
        )
    return scaled


def _token_text_key(text: str) -> str:
    value = re.sub(r"\s+", "", text or "")
    return value.upper()


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_centers_close(a: list[float], b: list[float]) -> bool:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    aw, ah = max(a[2] - a[0], 1.0), max(a[3] - a[1], 1.0)
    bw, bh = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
    return (
        abs(acx - bcx) <= max(aw, bw) * 0.45 + 12
        and abs(acy - bcy) <= max(ah, bh) * 0.60 + 10
    )


def _same_physical_token(left: OCRTokenData, right: OCRTokenData) -> bool:
    return _bbox_iou(left.bbox, right.bbox) >= 0.45 or _bbox_centers_close(left.bbox, right.bbox)


def _token_is_useful(token: OCRTokenData) -> bool:
    text = (token.text or "").strip()
    if not text:
        return False
    confidence = max(min(float(token.confidence or 0.0), 1.0), 0.0)
    if confidence >= 0.30:
        return True
    if _contains_cjk(text) and confidence >= 0.12:
        return True
    if confidence >= 0.15 and (
        re.search(r"\d+[,.]\d{2}", text)
        or re.search(r"\d{2,4}[-/]\d{1,2}[-/]\d{1,4}", text)
        or re.search(r"\d{6,}", text)
    ):
        return True
    return False


def _token_to_debug_dict(token: OCRTokenData) -> dict:
    return {
        "text": token.text,
        "bbox": [round(float(value), 2) for value in token.bbox],
        "confidence": round(float(token.confidence), 6),
        "page": token.page,
        "source_variant": token.source_variant,
        "signal_score": round(float(_token_signal_score(token)), 6),
    }


def _safe_debug_name(path: Path, engine: str, page_number: int) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "document"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_stem}_page{page_number}_{engine}_{timestamp}.json"


def _write_ocr_debug_report(
    *,
    engine: str,
    reference_path: Path,
    page_number: int,
    variant_records: list[dict],
    selected_tokens: list[OCRTokenData],
    selection_mode: str,
) -> None:
    if not _ocr_debug_enabled():
        return
    debug_dir = _ocr_debug_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": engine,
        "reference_image": str(reference_path),
        "page": page_number,
        "selection_mode": selection_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant_count": len(variant_records),
        "selected_metrics": _ocr_quality_metrics(selected_tokens),
        "selected_tokens": [_token_to_debug_dict(token) for token in selected_tokens],
        "variants": variant_records,
    }
    output_path = debug_dir / _safe_debug_name(reference_path, engine, page_number)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _fuse_ocr_tokens(token_sets: list[tuple[str, list[OCRTokenData]]]) -> list[OCRTokenData]:
    fused: list[tuple[OCRTokenData, int, set[str]]] = []
    for variant_name, tokens in token_sets:
        for token in tokens:
            if not _token_is_useful(token):
                continue
            key = _token_text_key(token.text)
            matched_idx: int | None = None
            for idx, (existing, _, _) in enumerate(fused):
                if key == _token_text_key(existing.text) and _same_physical_token(existing, token):
                    matched_idx = idx
                    break

            if matched_idx is None:
                fused.append((token, 1, {variant_name}))
                continue

            existing, support_count, variants = fused[matched_idx]
            variants.add(variant_name)
            best = token if token.confidence > existing.confidence else existing
            boosted_confidence = min(max(existing.confidence, token.confidence) + 0.035 * support_count, 0.99)
            fused[matched_idx] = (
                OCRTokenData(
                    text=best.text,
                    bbox=list(best.bbox),
                    confidence=boosted_confidence,
                    page=best.page,
                    source_variant="+".join(sorted(variants)),
                    source_engine=best.source_engine,
                ),
                support_count + 1,
                variants,
            )

    fused_tokens = [item[0] for item in fused]
    return sorted(fused_tokens, key=lambda token: (token.page, token.bbox[1], token.bbox[0], token.text))


def _paddle_variant_quality(tokens: list[OCRTokenData]) -> float:
    if any(_contains_cjk(token.text) for token in tokens):
        return _chinese_ocr_quality(tokens)
    return _ocr_quality(tokens)


def _best_paddle_tokens_for_variants(
    reader,
    variants: list[tuple[str, Path]],
    page_number: int,
    force_fusion: bool = False,
    single_pass: bool = False,
    timeout_s: int = 120,
) -> list[OCRTokenData]:
    reference_path = variants[0][1]
    raw_name, raw_path = variants[0]
    best_tokens = _scale_tokens_to_reference(
        _read_paddle_image(reader, raw_path, page_number, timeout_s=timeout_s),
        raw_name,
        raw_path,
        reference_path,
    )
    variant_records: list[dict] = [
        {
            "name": raw_name,
            "path": str(raw_path),
            "metrics": _ocr_quality_metrics(best_tokens),
            "tokens": [_token_to_debug_dict(token) for token in best_tokens],
        }
    ]

    if single_pass:
        _write_ocr_debug_report(
            engine="paddleocr",
            reference_path=reference_path,
            page_number=page_number,
            variant_records=variant_records,
            selected_tokens=best_tokens,
            selection_mode="single_pass",
        )
        return best_tokens

    should_try_variants = (
        _paddle_preprocessing_enabled()
        and (
            force_fusion
            or _average_confidence(best_tokens) < _low_confidence_threshold()
            or _tokens_look_like_chinese_invoice(best_tokens)
            or len(best_tokens) < 5
        )
    )
    if not should_try_variants:
        _write_ocr_debug_report(
            engine="paddleocr",
            reference_path=reference_path,
            page_number=page_number,
            variant_records=variant_records,
            selected_tokens=best_tokens,
            selection_mode="raw_only",
        )
        return best_tokens

    token_sets: list[tuple[str, list[OCRTokenData]]] = [(raw_name, best_tokens)]
    best_quality = _paddle_variant_quality(best_tokens)
    best_single_variant = best_tokens
    for variant_name, variant_path in variants[1:]:
        variant_tokens = _scale_tokens_to_reference(
            _read_paddle_image(reader, variant_path, page_number, timeout_s=timeout_s),
            variant_name,
            variant_path,
            reference_path,
        )
        token_sets.append((variant_name, variant_tokens))
        variant_records.append(
            {
                "name": variant_name,
                "path": str(variant_path),
                "metrics": _ocr_quality_metrics(variant_tokens),
                "tokens": [_token_to_debug_dict(token) for token in variant_tokens],
            }
        )
        quality = _paddle_variant_quality(variant_tokens)
        if quality > best_quality:
            best_single_variant = variant_tokens
            best_quality = quality
    if not _paddle_fusion_enabled():
        _write_ocr_debug_report(
            engine="paddleocr",
            reference_path=reference_path,
            page_number=page_number,
            variant_records=variant_records,
            selected_tokens=best_single_variant,
            selection_mode="best_single_variant",
        )
        return best_single_variant
    fused_tokens = _fuse_ocr_tokens(token_sets)
    _write_ocr_debug_report(
        engine="paddleocr",
        reference_path=reference_path,
        page_number=page_number,
        variant_records=variant_records,
        selected_tokens=fused_tokens,
        selection_mode="fused_variants",
    )
    return fused_tokens


def _run_paddleocr(path: Path, languages: tuple[str, ...]) -> list[OCRTokenData]:
    lang = _PADDLE_LANG_MAP.get((languages[0] if languages else "en"), "en")
    reader = _get_paddleocr_reader(lang)
    is_cjk = lang == "ch" or _has_chinese_language(tuple(languages))
    # CJK documents use the variant ensemble unless speed is explicitly preferred.
    timeout_s = _ocr_timeout()

    tokens: list[OCRTokenData] = []
    for page_number, page_path in enumerate(_image_paths(path), start=1):
        page_path = _cap_resolution(page_path, _ocr_max_dim())
        try:
            variants = preprocess_image_variants(page_path)
        except PreprocessingError as exc:
            raise OCRDependencyError(str(exc)) from exc
        variant_limit = _ocr_variant_limit()
        if variant_limit:
            variants = variants[:variant_limit]
        tokens.extend(_best_paddle_tokens_for_variants(
            reader,
            variants,
            page_number,
            force_fusion=_paddle_force_fusion_enabled(),
            single_pass=is_cjk and _paddle_cjk_single_pass_enabled(),
            timeout_s=timeout_s,
        ))
    return tokens


def _run_tesseract(path: Path) -> list[OCRTokenData]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRDependencyError("pytesseract and Pillow are required for Tesseract OCR.") from exc

    timeout_s = _ocr_timeout()
    tokens: list[OCRTokenData] = []
    for page_number, image_path in enumerate(_preprocessed_paths(path), start=1):
        image_path = _cap_resolution(image_path, _ocr_max_dim())

        def _infer(img_path=image_path):
            return pytesseract.image_to_data(Image.open(img_path), output_type=pytesseract.Output.DICT)

        data = _run_with_timeout(_infer, timeout_s, {"text": []})
        for idx, text in enumerate(data.get("text", [])):
            text = text.strip()
            if not text:
                continue
            x = float(data["left"][idx])
            y = float(data["top"][idx])
            w = float(data["width"][idx])
            h = float(data["height"][idx])
            try:
                confidence = max(float(data["conf"][idx]) / 100.0, 0.0)
            except ValueError:
                confidence = 0.0
            tokens.append(
                OCRTokenData(
                    text=text,
                    bbox=[x, y, x + w, y + h],
                    confidence=confidence,
                    page=page_number,
                    source_engine="tesseract",
                )
            )
    return tokens
