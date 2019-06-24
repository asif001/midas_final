from django.contrib import admin
from .models import DoctorDetails, PendingList, SubmittedList, Types

# Register your models here.
admin.site.register(DoctorDetails)
admin.site.register(PendingList)
admin.site.register(SubmittedList)
admin.site.register(Types)
