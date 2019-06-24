from django.db import models


# Create your models here.
class AssistantDetails(models.Model):
    userId = models.IntegerField(unique=True)
    name = models.CharField(max_length=25)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=50)
    isActive = models.BooleanField(default=False)

    def __str__(self):
        return self.name + "(" + str(self.userId) + ")"
