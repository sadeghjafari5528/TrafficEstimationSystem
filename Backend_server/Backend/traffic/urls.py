from django.urls import path

from traffic.views import predict_traffic, get_cars

app_name = 'traffic'

urlpatterns = [
    path('predict_traffic/', predict_traffic, name='predict_traffic'),
    path('get_cars/', get_cars, name='get_cars'),
]