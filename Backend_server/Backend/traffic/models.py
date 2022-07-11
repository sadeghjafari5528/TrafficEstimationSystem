from django.db import models

# Create your models here.

class VerificationCode(models.Model):
    date = models.DateField(_("Date"), default=datetime.now())
    no_cars = models.IntegerField()