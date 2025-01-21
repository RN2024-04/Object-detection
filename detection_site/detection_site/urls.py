from django.contrib import admin
from django.urls import path, include
from object_detection import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    path('admin/',admin.site.urls),
    path('register/', views.register),
    path('',include('object_detection.urls'))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# cd C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\detection_site
# python manage.py runserver

# ln -s C:/Users/asus/PycharmProject/UrbanProject/detection_site/db.sqlite3 trainval_lmdb
# C:/Users/asus/PycharmProject/UrbanProject/detection_site/db.sqlite3