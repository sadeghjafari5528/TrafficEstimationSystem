from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from traffic.models import Record
from traffic.RNN_model import model

@api_view(['GET'])
def predict_traffic(request):
    data = {"message":"predict_traffic"}
    return Response(data=data, status=status.HTTP_201_CREATED)



@api_view(['POST', ])
def get_cars(request):
    no_cars = request.data['no_cars']
    record = Record(no_cars=no_cars)
    record.save()
    data = {"message":"create record successfully"}
    return Response(data=data, status=status.HTTP_201_CREATED)