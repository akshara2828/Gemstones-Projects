from django.urls import path
from django.urls import re_path
from . import views

urlpatterns = [
    path('', views.home, name='app-home'),
    #path('about/',views.about,name='app-about'),
    path('form1/',views.form1,name = 'app-form1'),
    path('display/',views.display,name ='display')
    
]