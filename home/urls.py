from django.urls import path, include
from django.views.decorators.cache import never_cache
from . import views

app_name = 'home'

urlpatterns = [
    path('', never_cache(views.index), name='home'),
]
