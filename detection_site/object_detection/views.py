from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import  UserRegisterForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .forms import ImageUploadForm
from .models import ImageUpload
from PIL import Image as PILImage
import numpy as np
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
import os
from django.core.files.storage import default_storage
from django.conf import settings


def index2(request):
    return render(request, 'object_detection/home.html')


def login1(request):
    if request.method=='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
            # аутефикация
        user=authenticate(request,username=username,password=password)
        if user:
            login1(request,user)
            return redirect("http://127.0.0.1:8000/")
        else:
            return render(request, 'object_detection/login.html',{'error': 'Неверные учетные данные'})
    return render(request, 'object_detection/login.html')


def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse('user_login'))
    else:
        form = UserRegisterForm()
    return render(request, "object_detection/register.html", {"form": form})


# Загрузка модели
prototxt_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\voc\deploy.prototxt'
caffemodel_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\voc\mobilenet_iter_73000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


# Классы, которые может обнаружить модель
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person",
               "sheep", "sofa", "train", "tvmonitor"]


def process_image(image):
    try:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if idx < len(class_names):
                    label = f"{class_names[idx]}: {confidence:.2f}"
                else:
                    label = f"Unknown: {confidence:.2f}"

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def dashboard(request):
    fs = FileSystemStorage()
    uploaded_file_url = None
    output_file_url = None

    if request.method == 'POST':
        if 'image' in request.FILES:
            uploaded_file = request.FILES['image']
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_url = fs.url(filename)

            # Обработка изображения
            image_path = fs.path(filename)
            image = cv2.imread(image_path)

            if image is not None:
                output_image = process_image(image)  # Обработка изображения
                if output_image is not None:
                    # Сохранение обработанного изображения
                    output_filename = 'output_' + uploaded_file.name.split('.')[0] + '.jpg'
                    output_path = fs.path(output_filename)
                    cv2.imwrite(output_path, output_image)
                    output_file_url = fs.url(output_filename)
                else:
                    print("Error: Processed image is None.")
            else:
                print("Error: Loaded image is None.")

        elif 'delete' in request.POST:
            # Удаление загруженного изображения
            if 'image_name' in request.POST:
                image_name = request.POST['image_name']
                full_path = os.path.join(settings.MEDIA_ROOT, image_name)
                print(f"Full path to check: {full_path}")  # Отладочное сообщение
                if os.path.exists(full_path):
                    print("File exists.")
                else:
                    print("File does not exist.")

                output_image_name = 'output_' + image_name.split('.')[0] + '.jpg'
                fs.delete(output_image_name)  # Удаляем обработанное изображение

    # Извлечение имени файла
    uploaded_file_name = os.path.basename(uploaded_file_url) if uploaded_file_url else None

    return render(request, 'object_detection/dashboard.html', {
        'uploaded_file_url': uploaded_file_url,
        'output_file_url': output_file_url,
        'uploaded_file_name': uploaded_file_name,
    })




def logout1(request):
    logout(request)
    return redirect(reverse('user_login'))


