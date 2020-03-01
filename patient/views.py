from django.http import HttpResponsePermanentRedirect
from django.shortcuts import render
from django.urls import reverse

from .models import PatientDetails
from doctor.models import DoctorDetails, PendingList, SubmittedList
from assistant.models import AssistantDetails
from .forms import PatientLogin


# Create your views here.

# Patient home
def index(request):
    patient_set = PatientDetails.objects.all()
    for patient in patient_set:
        if patient.isActive:
            pending_set = PendingList.objects.filter(patientId=patient)
            submitted_set = SubmittedList.objects.filter(patientId=patient)
            context = {"patient": patient, "pending": pending_set, "submitted": submitted_set}
            return render(request, 'patient/pages/dashboard.html', context)

    return HttpResponsePermanentRedirect(reverse('patient:login'))


# Login
def login(request):
    # Check if already logged in
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

    # Get login Information
    if request.method == 'POST':
        form = PatientLogin(request.POST)

        if form.is_valid():
            for patient in patient_set:
                if patient.userId == form.cleaned_data['user_id'] and patient.password == \
                        form.cleaned_data['password']:
                    patient.isActive = True
                    patient.save()
                    return HttpResponsePermanentRedirect(reverse('patient:dashboard'))
        else:
            pass

    else:
        form = PatientLogin()

    return render(request, 'patient/registration/login.html', {'form': form})


# Logout
def logout(request):
    patient_set = PatientDetails.objects.all()
    for patient in patient_set:
        if patient.isActive:
            patient.isActive = False
            patient.save()
            return HttpResponsePermanentRedirect(reverse('home:home'))

    return HttpResponsePermanentRedirect(reverse('home:home'))
