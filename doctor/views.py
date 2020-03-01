from time import sleep

from django.http import HttpResponsePermanentRedirect, HttpResponseRedirect, HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import DoctorLogin, ReportForm
from .models import DoctorDetails, SubmittedList, PendingList
from assistant.models import AssistantDetails
from patient.models import PatientDetails

from classifiers.chest_X_ray_view_classifier import main

import os
import cv2
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Create your views here.

# Doctor Home
def index(request):
    doctor_set = DoctorDetails.objects.all()
    for doctor in doctor_set:
        if doctor.isActive:
            pending_set = PendingList.objects.filter(doctorId=doctor)
            submitted_set = SubmittedList.objects.filter(doctorId=doctor)
            context = {'doctor': doctor, 'pending': pending_set, 'submitted': submitted_set}
            # print(context)
            return render(request, 'doctor/pages/dashboard.html', context)

    return HttpResponsePermanentRedirect(reverse('doctor:login'))


# Doctor Login
def login(request):
    # Check Previous Login
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

    if request.method == 'POST':
        form = DoctorLogin(request.POST)

        if form.is_valid():
            for doctor in doctor_set:
                if doctor.userId == form.cleaned_data['user_id'] and doctor.password == form.cleaned_data['password']:
                    doctor.isActive = True
                    doctor.save()
                    return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))
        else:
            pass

    else:
        form = DoctorLogin()

    return render(request, 'doctor/registration/login.html', {'form': form})


# Doctor Logout
def logout(request):
    doctor_set = DoctorDetails.objects.all()
    for doctor in doctor_set:
        if doctor.isActive:
            doctor.isActive = False
            doctor.save()
            return HttpResponsePermanentRedirect(reverse('home:home'))

    doctor_set = PatientDetails.objects.all()
    for doctor in doctor_set:
        if doctor.isActive:
            doctor.isActive = False
            doctor.save()
            return HttpResponsePermanentRedirect(reverse('home:home'))

    doctor_set = AssistantDetails.objects.all()
    for doctor in doctor_set:
        if doctor.isActive:
            doctor.isActive = False
            doctor.save()
            return HttpResponsePermanentRedirect(reverse('home:home'))

    return HttpResponsePermanentRedirect(reverse('home:home'))


def getdirectorystuffs(pendingitem):
    IMAGE_DIR = "/".join(BASE_DIR.split("\\")) + str(pendingitem.images.url)
    IMAGE_DIR_ROOT = "/".join(str(pendingitem.images.url).split("/")[0:-1])
    DIR_ROOT = "/".join(IMAGE_DIR.split("/")[0:-1])
    IMAGE_NAME = str(pendingitem.images.name).split("/")[-1]
    TEMP_NAME = "temp_" + str(pendingitem.images.name).split("/")[-1]
    TEMP_IMAGE_DIR = "/".join(BASE_DIR.split("\\")) + IMAGE_DIR_ROOT + "/" + TEMP_NAME
    TEMP_IMAGE_SRC = IMAGE_DIR_ROOT + "/" + TEMP_NAME

    return {"IMAGE_DIR": IMAGE_DIR, "IMAGE_DIR_ROOT": IMAGE_DIR_ROOT, "TEMP_NAME": TEMP_NAME,
            "TEMP_IMAGE_DIR": TEMP_IMAGE_DIR, "TEMP_IMAGE_SRC": TEMP_IMAGE_SRC, "IMAGE_NAME": IMAGE_NAME,
            "DIR_ROOT": DIR_ROOT}


# Analysis The Patient's Pending Item
def analysis(request, pendingid):
    if request.method == 'POST':
        form = ReportForm(request.POST)
        if form.is_valid():
            pendingitem = PendingList.objects.get(id=pendingid)
            submitteditem = SubmittedList()
            submitteditem.doctorId, submitteditem.patientId, submitteditem.type = pendingitem.doctorId, pendingitem.patientId, pendingitem.type
            NEW_DIR = "/".join(str(pendingitem.images.url).replace("pending", "submitted").split("/")[2:])
            CUR_DIR = getdirectorystuffs(pendingitem)["DIR_ROOT"].replace("pending", "submitted")
            submitteditem.image = NEW_DIR
            submitteditem.report = form.cleaned_data['report']
            if os.path.exists(CUR_DIR):
                shutil.rmtree(CUR_DIR)
            os.makedirs(CUR_DIR)
            shutil.move(getdirectorystuffs(pendingitem)["IMAGE_DIR"], CUR_DIR)
            os.remove(getdirectorystuffs(pendingitem)["TEMP_IMAGE_DIR"])
            submitteditem.save()
            pendingitem.delete()

            return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

        else:

            return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    try:
        pendingitem = PendingList.objects.get(id=pendingid)
    except:
        return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    form = ReportForm()
    context = {'pendingItem': pendingitem, 'form': form}

    # READ WRITE IMAGE #
    img = cv2.imread(getdirectorystuffs(pendingitem)["IMAGE_DIR"], 0)
    cv2.imwrite(getdirectorystuffs(pendingitem)["TEMP_IMAGE_DIR"], img)

    # print(pendingitem.images.url)
    if pendingitem.type.name in ["chest", "Pneumonia"]:
        return render(request, 'doctor/pages/chestanalysis.html', context)
    elif pendingitem.type.name in ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']:
        return render(request, 'doctor/pages/msd.html', context)
    elif pendingitem.type.name in ["mammography"]:
        return render(request, 'doctor/pages/mammography.html', context)

    return HttpResponse("In Analysys")


# Return The Result After Analysis
def result(request):
    if request.method == "GET":
        pending_id = int(request.GET.get('id'))
        type = str(request.GET.get('type'))
    else:
        return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    try:
        pendingitem = PendingList.objects.get(id=pending_id)
    except:
        return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    if type in ["chest", "Pneumonia"]:
        context = {'img_1.jpg': ['comment1.txt', 'report_1.jpg'], 'img_2.jpg': ['comment2.txt', 'report_2.jpg'],
                   'img_3.jpg': ['comment3.txt', 'report_3.jpg'], 'img_4.jpg': ['comment4.txt', 'report_4.jpg']}

        if getdirectorystuffs(pendingitem)['IMAGE_NAME'] in context:
            report = open("/home/asifurarahman/midas_final/media/classifiers/pneumonia/"
                          + context[getdirectorystuffs(pendingitem)['IMAGE_NAME']][0], "r+")

            final_report = ""

            for line in report:
                final_report = final_report + line + "<br/>"

            sleep(1.5)
            return HttpResponse(final_report+"?"+context[getdirectorystuffs(pendingitem)['IMAGE_NAME']][1])
    elif type in ["position"]:
        result = main.view_predict(getdirectorystuffs(pendingitem)["IMAGE_DIR"])
        return HttpResponse(result)
    elif type in ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']:
        context = {"abnormal_XR_ELBOW_hist_276.png": ['Abnormal', 'Elbow_Result_of_VGG_Model1.txt', 'XR_ELBOW'],
                   "normal_XR_ELBOW_hist_13.png": ['Normal', 'Elbow_Result_of_VGG_Model2.txt', 'XR_ELBOW'],
                   "abnormal_XR_FINGER_hist_422.png": ['Abnormal', 'Finger_Result_of_VGG_Model1.txt', 'XR_FINGER'],
                   "normal_XR_FINGER_hist_82.png": ['Normal', 'Finger_Result_of_VGG_Model2.txt', 'XR_FINGER'],
                   "abnormal_XR_FOREARM_hist_231.png": ['Abnormal', 'Forearm_Result_of_VGG_Model1.txt', 'XR_FOREARM'],
                   "normal_XR_FOREARM_hist_109.png": ['Normal', 'Forearm_Result_of_VGG_Model.txt2', 'XR_FOREARM'],
                   "abnormal_XR_HAND_hist_348.png": ['Abnormal', 'Hand_Result_of_VGG_Model.txt1', 'XR_HAND'],
                   "normal_XR_HAND_hist_155.png": ['Normal', 'Hand_Result_of_VGG_Model2.txt', 'XR_HAND'],
                   "abnormal_XR_HUMERUS_hist_267.png": ['Abnormal', 'Humerus_Result_of_VGG_Model1.txt', 'XR_HUMERUS'],
                   "normal_XR_HUMERUS_hist_68.png": ['Normal', 'Humerus_Result_of_VGG_Model2.txt', 'XR_HUMERUS'],
                   "abnormal_XR_SHOULDER_hist_240.png": ['Abnormal', 'Shoulder_Result_of_VGG_Model1.txt', 'XR_SHOULDER'],
                   "normal_XR_SHOULDER_hist_13.png": ['Normal', 'Shoulder_Result_of_VGG_Model2.txt', 'XR_SHOULDER'],
                   "abnormal_XR_WRIST_hist_422.png": ['Abnormal', 'Wrist_Result_of_VGG_Model1.txt', 'XR_WRIST'],
                   "normal_XR_WRIST_hist_64.png": ['Normal', 'Wrist_Result_of_VGG_Model2.txt', 'XR_WRIST']}

        if getdirectorystuffs(pendingitem)['IMAGE_NAME'] in context:
            report = open("/home/asifurarahman/midas_final/midas_final/media/classifiers/Musculoskeletal/" +
                          context[getdirectorystuffs(pendingitem)['IMAGE_NAME']][2] + "/" +
                          context[getdirectorystuffs(pendingitem)['IMAGE_NAME']][1], "r+")

            msd_type = context[getdirectorystuffs(pendingitem)['IMAGE_NAME']][0]

            final_report = ""

            for line in report:
                final_report = final_report + line + "<br/>"

            sleep(0.5)
            return HttpResponse(final_report + "?" + msd_type)

    elif type in ["mammography"]:
        context = {"benign1.png": ["Benign"], "benign2.png": ["Benign"],
                   "InSitu1.png": ["Insitu"], "InSitu2.png": ["Insitu"],
                   "Invasive1.png": ["Invasive"], "Invasive2.png": ["Invasive"],
                   "Normal1.png": ["Normal"], "Normal2.png": ["Normal"]}

        report = ""

        if getdirectorystuffs(pendingitem)['IMAGE_NAME'] in context:
            report = context[getdirectorystuffs(pendingitem)['IMAGE_NAME']][0]

        sleep(0.5)
        return HttpResponse(report)

    return HttpResponse("OK")


def submitted_item(request, submittedid):
    try:
        submitteditem = SubmittedList.objects.get(id=submittedid)
    except:
        return HttpResponsePermanentRedirect(reverse('doctor:dashboard'))

    context = {"submitted": submitteditem}

    return render(request, 'doctor/pages/submitted.html', context)


def change_threshold(request):
    pendingid = int(request.GET.get('id'))
    minimum = int(request.GET.get('minimum'))
    maximum = int(request.GET.get('maximum'))
    pendingitem = PendingList.objects.get(id=pendingid)

    # Thresholding
    img = cv2.imread(getdirectorystuffs(pendingitem)["IMAGE_DIR"], 0)
    mask = (img >= minimum) & (img <= maximum)
    img = img * mask
    cv2.imwrite(getdirectorystuffs(pendingitem)["TEMP_IMAGE_DIR"], img)

    return HttpResponse(getdirectorystuffs(pendingitem)["TEMP_IMAGE_SRC"])


def change_contrast(request):
    pendingid = int(request.GET.get('id'))
    alpha = float(request.GET.get('alpha'))
    beta = int(request.GET.get('beta'))
    pendingitem = PendingList.objects.get(id=pendingid)

    # print(pendingid)

    img = cv2.imread(getdirectorystuffs(pendingitem)["IMAGE_DIR"], 0)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite(getdirectorystuffs(pendingitem)["TEMP_IMAGE_DIR"], img)

    return HttpResponse(getdirectorystuffs(pendingitem)["TEMP_IMAGE_SRC"])


def detect_edge(request):
    pendingid = int(request.GET.get('id'))
    pendingitem = PendingList.objects.get(id=pendingid)

    # print(pendingid)
    img = cv2.imread(getdirectorystuffs(pendingitem)["IMAGE_DIR"], 0)
    img = cv2.Canny(img, 10, 20)
    cv2.imwrite(getdirectorystuffs(pendingitem)["TEMP_IMAGE_DIR"], img)

    return HttpResponse(getdirectorystuffs(pendingitem)["TEMP_IMAGE_SRC"])



def equalize_hist(request):
    pendingid = int(request.GET.get('id'))
    pendingitem = PendingList.objects.get(id=pendingid)

    # print(pendingid)
    img = cv2.imread(getdirectorystuffs(pendingitem)["IMAGE_DIR"], 0)
    img = cv2.equalizeHist(img)
    cv2.imwrite(getdirectorystuffs(pendingitem)["TEMP_IMAGE_DIR"], img)

    return HttpResponse(getdirectorystuffs(pendingitem)["TEMP_IMAGE_SRC"])


def print_report(request):
    return render(request, 'doctor/pages/print.html')
