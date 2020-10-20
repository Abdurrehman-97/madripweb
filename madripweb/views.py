from django.shortcuts import render
import requests
from subprocess import run,PIPE
import sys
from django.core.files.storage import FileSystemStorage

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
    out = run([sys.executable,'C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\verify.py',option,name,email,org,phn,uname,pwd],shell=False,stdout=PIPE)
    print(out)

    return render(request, 'Intermediary.html',{'data1' : out.stdout})

def Signin(request):
    
    option = "L"
    uname=request.POST.get('uname')
    pwd=request.POST.get('pass')
    out = run([sys.executable,'C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\verify.py',option,uname,pwd],shell=False,stdout=PIPE)
    print(out)
    return render(request, 'Intermediary.html',{"data2" : out.stdout})

def RetinaUpload(request):
    return render(request,'RetinaScanUpload.html')

def ProcessUpload(request):
    scan=request.FILES['fileUpload']
    print("Image upload: ",scan)
    store=FileSystemStorage()
    f_name=store.save(scan.name,scan)
    f_url=store.open(f_name)
    #temp_url=str(f_url) - str(f_name)
    print("file raw url: ", f_name)
    print("file full url: ",f_url)
    #print("temp url: ",temp_url)
    out = run([sys.executable,'C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\process.py',f_name],shell=False,stdout=PIPE)
    return render(request,'Home.html')