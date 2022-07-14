from django.db import models
from datetime import datetime

# Create your models here.

class Record(models.Model):
    date = models.DateTimeField(default=datetime.now())
    no_cars = models.IntegerField()