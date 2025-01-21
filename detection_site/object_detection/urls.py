from django.urls import path
from . import views


urlpatterns=[
    path('',views.index2),
    path('login/', views.login1),
    path('home/',views.index2),
    path('register/', views.register),
    path('dashboard/', views.dashboard),
    path('logout/', views.logout1),
]

# cd C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\detection_site
# python manage.py runserver