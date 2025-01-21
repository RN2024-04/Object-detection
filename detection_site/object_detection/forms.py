from django import forms
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.contrib.auth.models import User
from .models import ImageUpload


class LoginForm (AuthenticationForm):
    username=forms.CharField()
    password = forms.CharField()

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)  # Добавляем поле email

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']  # Используем стандартные поля

    def save(self, commit=True):
        user = super().save(commit=False)  # Получаем пользователя из родительского класса
        user.email = self.cleaned_data['email']  # Сохраняем email
        if commit:
            user.save()  # Сохраняем пользователя
        return user


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ['image']