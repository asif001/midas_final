from django.http import HttpResponsePermanentRedirect
from django.shortcuts import render
from django.urls import reverse

from .models import AssistantDetails
from doctor.models import DoctorDetails
from patient.models import PatientDetails
from .forms import AssistantLogin, OperatorForm


# Create your views here.

# Assistant home
def index(request):
    if request.method == "POST":
        form = OperatorForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return HttpResponsePermanentRedirect(reverse('assistant:dashboard'))

    assistant_set = AssistantDetails.objects.all()
    for assistant in assistant_set:
        if assistant.isActive:
            form = OperatorForm()
            context = {"assistant": assistant, "form": form}
            return render(request, 'assistant/pages/dashboard.html', context)

    return HttpResponsePermanentRedirect(reverse('assistant:login'))


#Assistant Login
def login(request):
    # Check Login
    doctor_set = DoctorDetails.objects.all()
    for doctor in doctor_set:
        if doctor.isActive:
            return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    patient_set = PatientDetails.objects.all()
    for patient in patient_set:
        if patient.isActive:
            return HttpResponsePermanentRedirect(reverse('patient:dashboard'))

    assistant_set = AssistantDetails.objects.all()
    for assistant in assistant_set:
        if assistant.isActive:
            return HttpResponsePermanentRedirect(reverse('assistant:dashboard'))

    if request.method == 'POST':
        form = AssistantLogin(request.POST)

        if form.is_valid():
            for assistant in assistant_set:
                if assistant.userId == form.cleaned_data['user_id'] and assistant.password == \
                        form.cleaned_data['password']:
                    assistant.isActive = True
                    assistant.save()
                    return HttpResponsePermanentRedirect(reverse('assistant:dashboard'))
        else:
            print("Not working")

    else:
        form = AssistantLogin()

    return render(request, 'assistant/registration/login.html', {'form': form})


#Assistant Logout
def logout(request):
    assistant_set = AssistantDetails.objects.all()
    for assistant in assistant_set:
        if assistant.isActive:
            assistant.isActive = False
            assistant.save()
            return HttpResponsePermanentRedirect(reverse('home:home'))

    return HttpResponsePermanentRedirect(reverse('home:home'))
