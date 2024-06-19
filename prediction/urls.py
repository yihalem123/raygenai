from django.urls import path
from .views import predict_stock

urlpatterns = [
    path('predict/', predict_stock, name='predict_stock'),
]
