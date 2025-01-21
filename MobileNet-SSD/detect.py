import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Загрузка модели
prototxt_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\voc\deploy.prototxt'
caffemodel_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\voc\mobilenet_iter_73000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


# Классы, которые может обнаружить модель
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


# Загрузка изображения
image_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\create_lmdb\Dataset\Images\3.jpg'
image = cv2.imread(image_path)

(h, w) = image.shape[:2]

# Подготовка изображения для модели
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)

# Получение предсказаний
detections = net.forward()


# Обработка результатов
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.2:  # Порог уверенности
        idx = int(detections[0, 0, i, 1])

        # Проверяем, что индекс в пределах допустимого диапазона
        if idx < len(class_names):
            label = f"{class_names[idx]}: {confidence:.2f}"  # Используем имя класса
        else:
            label = f"Unknown: {confidence:.2f}"

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Убираем эту строку, так как она переопределяет label
        # label = f"Object {idx}: {confidence:.2f}"

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')  # Отключение осей
plt.show()

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# if image is None:
#     raise ValueError(f"Unable to load image at {image_path}")
# else:
# # Отображение изображения с результатами
#     plt.imshow(image_rgb)
#     plt.axis('off')  # Отключение осей
#     plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Загрузка модели
# prototxt_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\voc\MobileNetSSD_deploy.prototxt'
# caffemodel_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\voc\mobilenet_iter_73000.caffemodel'
#
# net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
#
# # Классы, которые может обнаружить модель
# class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
#                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#                "dog", "horse", "motorbike", "person", "pottedplant",
#                "sheep", "sofa", "train", "tvmonitor"]
#
# # Загрузка изображения
# image_path = r'C:\Users\AMCV\PycharmProjects\UrbanProject\UrbanProject\MobileNet-SSD\images\input_image.jpg'
# image = cv2.imread(image_path)
# (h, w) = image.shape[:2]
#
#
# # Подготовка изображения для нейронной сети
# blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
# net.setInput(blob)
#
# # Выполнение детекции
# detections = net.forward()
#
# # Обработка результатов
# for i in range(detections.shape[2]):
#     confidence = detections[0, 0, i, 2]
#     if confidence > 0.2:  # Порог уверенности
#         idx = int(detections[0, 0, i, 1])
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#
#         label = f"{class_names[idx]}: {confidence:.2f}"
#         cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#         cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
# # Отображение результата
# if image is not None:
#     # Преобразование BGR в RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Отображение изображения
#     plt.imshow(image_rgb)
#     plt.axis('off')  # Отключить оси
#     plt.show()
# else:
#     print("Не удалось загрузить изображение.")