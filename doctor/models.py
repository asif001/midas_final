from django.db import models
from patient.models import PatientDetails

# Create your models here.


class DoctorDetails(models.Model):
    userId = models.IntegerField(unique=True)
    name = models.CharField(max_length=25)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=50)
    isActive = models.BooleanField(default=False)

    def __str__(self):
        return self.name + "(" + str(self.userId) + ")"


class Types(models.Model):
    name = models.CharField(max_length=25)

    def __str__(self):
        return str(self.name)


def content_file_name_pending(instance, filename):
    return '/'.join(["images", str(instance.doctorId.userId), "pending", str(instance.patientId), str(instance.type),
                     str(filename)])


def content_file_name_submitted(instance, filename):
    return '/'.join(["images", str(instance.doctorId.userId), "submitted", str(instance.patientId), str(instance.type),
                     str(filename)])


class PendingList(models.Model):
    doctorId = models.ForeignKey(DoctorDetails, on_delete=models.CASCADE)
    patientId = models.ForeignKey(PatientDetails, on_delete=models.CASCADE)
    type = models.ForeignKey(Types, on_delete=models.CASCADE)
    images = models.ImageField(upload_to=content_file_name_pending)
    date = models.DateTimeField(auto_now_add=True)


class SubmittedList(models.Model):
    doctorId = models.ForeignKey(DoctorDetails, on_delete=models.CASCADE)
    patientId = models.ForeignKey(PatientDetails, on_delete=models.CASCADE)
    type = models.ForeignKey(Types, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=content_file_name_submitted)
    report = models.CharField(max_length=250)
    date = models.DateTimeField(auto_now_add=True)
