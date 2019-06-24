from django import forms
from doctor.models import PendingList


class AssistantLogin(forms.Form):
    user_id = forms.IntegerField(label="User Id", required=True, min_value=1)
    password = forms.CharField(label='Password', required=True, widget=forms.PasswordInput, min_length=8)


class OperatorForm(forms.ModelForm):
    class Meta:
        model = PendingList
        fields = ['doctorId', 'patientId', 'type', 'images']
