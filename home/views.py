from django.http import HttpResponsePermanentRedirect
from django.shortcuts import render
from django.urls import reverse

from doctor.models import DoctorDetails
from assistant.models import AssistantDetails
from patient.models import PatientDetails


# Create your views here.
def index(request):
    doctor_set = DoctorDetails.objects.all()
    for doctor in doctor_set:
        if doctor.isActive:
            return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    assistant_set = AssistantDetails.objects.all()
    for assistant in assistant_set:
        if assistant.isActive:
            return HttpResponsePermanentRedirect(reverse('assistant:dashboard'))

    patient_set = PatientDetails.objects.all()
    for patient in patient_set:
        if patient.isActive:
            return HttpResponsePermanentRedirect(reverse('patient:dashboard'))

    return render(request, 'home/pages/index.html')
