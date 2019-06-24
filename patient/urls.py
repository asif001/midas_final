from django.urls import path, include
from django.views.decorators.cache import never_cache
from . import views

app_name = 'patient'

urlpatterns = [
    path('accounts/login/', never_cache(views.login), name='login'),
    path('accounts/logout/', never_cache(views.logout), name='logout'),
    path('dashboard/', never_cache(views.index), name='dashboard'),
]
