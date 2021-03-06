"""madripweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#contains urls to be used for transitioning between pages..
#links the functions defined in views.py with urls..
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home,name='Home'),
    path('Login/', views.Login,name='Login'),
    path('Signin/', views.Signin,name='Signin'),
    path('imageupload/', views.imageupload,name='imageupload'),
    path('Signup/', views.Signup,name='Signup'),
    path('Register/', views.Register,name='Register'),
    path('RetinaUpload/', views.RetinaUpload,name='RetinaUpload'),
    path('ProcessUpload/', views.ProcessUpload,name='ProcessUpload'),
    path('UserOptions/', views.UserOptions,name='UserOptions'),
    path('UserInfo/', views.UserInfo,name='UserInfo'),
    path('Results/',views.IdentifyUpload,name="Results"),
    path('ExtractFeatures/', views.ExtractFeatures,name='ExtractFeatures'),
    path('ResultDME/', views.IdentifyDME,name='ResultDME'),
    path('Report/', views.GetReport,name='GetReport'),
    path('Help/', views.Help,name='Help'),
    path('FAQ/', views.FAQ,name='FAQ'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

