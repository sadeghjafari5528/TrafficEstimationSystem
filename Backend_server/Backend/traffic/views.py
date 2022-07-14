from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import numpy as np
from django.core.cache import cache
import tensorflow as tf

from traffic.models import Record
from traffic.RNN_model import window, get_series_data, scaler

@api_view(['GET'])
def predict_traffic(request):
    last_data = cache.get('last_data')
    
    test_data = last_data[-window:]
    # reshape data
    test_data = np.array(last_data).reshape(-1, 1)
    # normalize data
    X_test, Y_test = get_series_data(window, test_data)

    model = tf.keras.models.load_model('model')
    Y1_test = model.predict(X_test)
    Y1_test = scaler.inverse_transform(Y1_test)
    Y1_test = Y1_test.reshape(Y1_test.shape[0])
    data = {"message":"predict_traffic", "data":Y1_test}
    return Response(data=data, status=status.HTTP_201_CREATED)



@api_view(['POST', ])
def get_cars(request):
    no_cars = request.data['no_cars']
    record = Record(no_cars=no_cars)
    record.save()

    last_data = cache.get('last_data')
    last_data.insert(0, int(no_cars))
    last_data.pop()
    cache.set('last_data', last_data)
    

    # # reshape data
    # last_data = np.array(last_data).reshape(-1, 1)
    # # normalize data
    # X_train, Y_train = get_series_data(window, last_data)

    # model = tf.keras.models.load_model('model')
    # model.fit(
    #     X_train, Y_train, validation_split=0, batch_size=4, epochs=2
    # )
    # model.save('model')
    print("number of cars:", no_cars)
    data = {"message":"create record successfully"}
    return Response(data=data, status=status.HTTP_201_CREATED)