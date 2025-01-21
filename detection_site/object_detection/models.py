from django.db import models
from django.contrib.auth.models import AbstractUser

class PeopleReg(models.Model):
    username=models.CharField(max_length=150, default='')
    password=models.CharField(max_length=150, default='')
    password2=models.CharField(max_length=150, default='')
    # USERNAME_FIELD = 'username'
    def save(self, *args, **kwargs):
        # this will take care of the saving
        super(PeopleReg, self).save(*args, **kwargs)


class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result = models.CharField(max_length=255, blank=True)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name

