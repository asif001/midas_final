from django import forms


class PatientLogin(forms.Form):
    user_id = forms.IntegerField(label="User Id", required=True, min_value=1)
    password = forms.CharField(label='Password', required=True, widget=forms.PasswordInput, min_length=8)

