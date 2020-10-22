from django.shortcuts import render
import requests
from subprocess import run,PIPE
import sys
from django.core.files.storage import FileSystemStorage
import os
from . import settings
from django.utils.safestring import mark_safe



def home(request):
    return render(request, 'Home.html')

# def output(request):
#     data=requests.get("https://www.google.com.pk")
#     print(data.text)
#     data=data.text
#     return render(request, 'Home.html',{'data':data})

def Login(request):
    return render(request,'Login.html')
    
def imageupload(request):
    return render(request,'imageupload.html')

def Signup(request):
    return render(request,'Signup.html')
    
def Register(request):
    option = "R"
    name = request.POST.get('first_name') + " " + request.POST.get('last_name')
    email = request.POST.get('email')
    org = request.POST.get('organization')
    phn = request.POST.get('code') + request.POST.get('phone')
    uname=request.POST.get('uname')
    pwd=request.POST.get('pass')
    out = run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\verify.py'),option,name,email,org,phn,uname,pwd],shell=False,stdout=PIPE)
    print(out)

    return render(request, 'Intermediary.html',{'data1' : out.stdout})

def Signin(request):
    option = "L"
    uname=request.POST.get('uname')
    global u_name
    u_name=uname
    pwd=request.POST.get('pass')
    out = run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\verify.py'),option,uname,pwd],shell=False,stdout=PIPE)
    print(out)
    out = out.stdout
    result = out.decode('utf-8')
    return render(request, 'Intermediary.html',{"data2" : mark_safe(result),"uname" : u_name})

def RetinaUpload(request):
    return render(request,'RetinaScanUpload.html',{"uname": u_name})

def ProcessUpload(request):
    option="P"
    scan=request.FILES['fileUpload']
    print("Image upload: ",scan)
    store=FileSystemStorage()
    f_name=store.save(scan.name,scan)
    global file_name
    file_name=f_name
    f_url=store.open(f_name)
    #temp_url=str(f_url) - str(f_name)
    print("file raw url: ", f_name)
    print("file full url: ",f_url)
    #print("temp url: ",temp_url)
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),f_name,option],shell=False,stdout=PIPE)
    print("------------",dir)
    print("-------------",u_name)
    return render(request,'UserOptions.html',{"uname": u_name})

def UserOptions(request):
    return render(request,'UserOptions.html',{"uname": u_name})

def IdentifyUpload(request):
    option = "I"
    print("File name..........",file_name)
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),file_name,option],shell=False,stdout=PIPE)
    print("------------",dir)
    out = dir.stdout
    result = out.decode('utf-8')
    return render(request,'Results.html',{"result": mark_safe(result),"uname" : u_name})

def UserInfo(request):
    option="I"
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\verify.py'),option,u_name],shell=False,stdout=PIPE)
    print(".........User...",dir)
    out = dir.stdout
    result = out.decode('utf-8')
    print(result)
    return render(request,'UserInfo.html',{"result": mark_safe(result),"uname" : u_name})


def ExtractFeatures(request):
    option="E"
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),file_name,option],shell=False,stdout=PIPE)
    print(".........Result file",file_name)
    out = dir.stdout
    result = out.decode('utf-8')
    print(mark_safe(result))
    return render(request,'ExtractFeatures.html',{"result_image": mark_safe(result),"uname" : u_name})
   