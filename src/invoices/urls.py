from django.urls import path

from . import views


app_name = "invoices"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_document, name="upload"),
    path("documents/<int:pk>/", views.document_detail, name="detail"),
    path("documents/<int:pk>/process/", views.process_document, name="process"),
    path("documents/<int:pk>/review/", views.review_document, name="review"),
    path("documents/<int:pk>/export/<str:fmt>/", views.export_document, name="export"),
]

