from django.urls import path, include
from django.views.decorators.cache import never_cache
from . import views

app_name = 'doctor'

urlpatterns = [
    path('accounts/login/', never_cache(views.login), name='login'),
    path('accounts/logout/', never_cache(views.logout), name='logout'),
    path('dashboard/', never_cache(views.index), name='dashboard'),
    path('submitted/<int:submittedid>', never_cache(views.submitted_item), name='submitted'),
    path('analysis/<int:pendingid>', never_cache(views.analysis), name='analysis'),
    path('analysis/result', never_cache(views.result), name="result"),
    path('analysis/eqHist', never_cache(views.equalize_hist), name="eq_hist"),
    path('analysis/detectEdge', never_cache(views.detect_edge), name="detect_edge"),
    path('analysis/changeThreshold', never_cache(views.change_threshold), name='changethreshold'),
    path('analysis/changeContrast', never_cache(views.change_contrast), name='changecontrast'),
]
