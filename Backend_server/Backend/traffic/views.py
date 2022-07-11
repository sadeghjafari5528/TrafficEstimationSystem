from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

@api_view(['POST'])
def predict_traffic(request):
    data = {"message":"predict_traffic"}
    return Response(data=data, status=status.HTTP_201_CREATED)



@api_view(['GET', ])
def get_cars(request):
    data = {"message":"get_cars"}
    return Response(data=data, status=status.HTTP_201_CREATED)