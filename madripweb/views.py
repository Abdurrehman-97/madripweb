from django.shortcuts import render
import requests
from subprocess import run,PIPE
import sys

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

    return render(request, 'Home.html',{'data1' : out.stdout})

def Signin(request):
    
    option = "L"
    uname=request.POST.get('uname')
    pwd=request.POST.get('pass')
    out = run([sys.executable,'C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\verify.py',option,uname,pwd],shell=False,stdout=PIPE)
    print(out)
    return render(request, 'Home.html',{"data2" : out.stdout})