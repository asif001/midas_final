from django import forms


class DoctorLogin(forms.Form):
    user_id = forms.IntegerField(label="User Id", required=True, min_value=1)
    password = forms.CharField(label='Password', required=True, widget=forms.PasswordInput, min_length=8)


class ReportForm(forms.Form):
    report = forms.CharField(widget=forms.Textarea(attrs={'cols': 80, 'rows': 5}))
