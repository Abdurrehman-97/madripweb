#Contains functions that run in the background returns a respective HTML page........
#also runs the backend code present in process.py and verify.py....

from django.shortcuts import render
import requests
from subprocess import run,PIPE
import sys
from django.core.files.storage import FileSystemStorage
import os
from . import settings
from django.utils.safestring import mark_safe

#global variables

u_name = None
patient_id = None
file_name = None
U_name = "xyz"
option =""
retina_stage = None
edema_stage = None
feature1 = None
feature2 = None

def home(request):
    return render(request, 'Home.html')


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

    #registration code being run by verify.py..
    out = run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\verify.py'),option,name,email,org,phn,uname,pwd],shell=False,stdout=PIPE)
    print(out)

    return render(request, 'Intermediary.html',{'data1' : out.stdout})

def Signin(request):
    option = "L"
    uname=request.POST.get('uname')
    global u_name
    u_name=uname
    pwd=request.POST.get('pass')

    #verification code being run by verify.py..
    out = run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\verify.py'),option,uname,pwd],shell=False,stdout=PIPE)
    print(out)
    out = out.stdout
    result = out.decode('utf-8')
    if 'T' in result:
        option = "S"
        return render(request, 'RetinaScanUpload.html',{"data2" : mark_safe(result),"uname" : u_name,"option": option})
    else:
        option = "A"
        return render(request, "Login.html",{'option':option})

def RetinaUpload(request):
    return render(request,'RetinaScanUpload.html',{"uname": u_name})

def ProcessUpload(request):
    option="P"
    global patient_id
    patient_id=request.POST.get('P_id')
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

    #pre-processing code to be run by process.py..
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),f_name,option],shell=False,stdout=PIPE)
    print("------------",dir)
    print("-------------",u_name)
    global path
    path = '\\scans\\' + file_name   
    out = dir.stdout
    result =  out.decode('utf-8')
    if 'T' in result:
        option = "S"
        return render(request,'UserOptions.html',{"uname": u_name, "subject_image":path,"option":option,"patient_id":patient_id})
    else:
        option ="A"
        return render(request,'RetinaScanUpload.html',{"uname": u_name,"option":option})

def UserOptions(request):
    return render(request,'UserOptions.html',{"uname": u_name,"subject_image":path})

def IdentifyUpload(request):
    option = "R"
    print("File name..........",file_name)

    #DR identification code in process.py, file name passed as argument
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),file_name,option],shell=False,stdout=PIPE)
    print("------------",dir)
    out = dir.stdout
    
    result = out.decode('utf-8')
    print("Results -- ", result)
    global retina_stage
    retina_stage = result
    if retina_stage != None:
        option ="S"
        return render(request,'Results.html',{"DR": mark_safe(result),"uname" : u_name,"option":option})
    else:
        return render(request,'Results.html',{"DR": mark_safe(result),"uname" : u_name})

def UserInfo(request):
    option="I"

    #user information fetched from Database..
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\verify.py'),option,u_name],shell=False,stdout=PIPE)
    print(".........User...",dir)
    out = dir.stdout
    result = out.decode('utf-8')
    print(result)
    return render(request,'UserInfo.html',{"result": mark_safe(result),"uname" : u_name})


def ExtractFeatures(request):
    option="E"

    #Feature extraction from image, file name passed as argument..
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),file_name,option],shell=False,stdout=PIPE)
    print(".........Result file",file_name)
    out = dir.stdout
    result = out.decode('utf-8')
    print("E-result1",mark_safe(result))
    result2 = '\\static\\BloodVImage\\' + file_name
    global feature1
    feature1 = result
    global feature2
    feature2 = result2
    if feature1 != None:
        option="S"
        return render(request,'ExtractFeatures.html',{"exudate_image": mark_safe(result),"uname" : u_name, "blood_image" : result2, "option": option})
    else:
        return render(request,'ExtractFeatures.html',{"exudate_image": mark_safe(result),"uname" : u_name, "blood_image" : result2})

def IdentifyDME(request):
    option = "M"
    print("File name..........",file_name)

    #DME identification code in process.py, file name passed as argument
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),file_name,option],shell=False,stdout=PIPE)
    print("------------",dir)
    out = dir.stdout
    result = out.decode('utf-8')
    global edema_stage
    edema_stage = result
    print("Results -- ", result)
    if edema_stage != None:
        option ="S"
        return render(request,'ResultDME.html',{"DME": mark_safe(result),"uname" : u_name,"option":option})
    else:
        return render(request,'ResultDME.html',{"DME": mark_safe(result),"uname" : u_name})

def GetReport(request):
    option = "G"

    #Report generation code present in another file, file name passed as argument..
    dir=run([sys.executable,os.path.join(settings.BASE_DIR, 'madripweb\\process.py'),file_name,option,retina_stage,edema_stage,feature1,feature2,patient_id],shell=False,stdout=PIPE)
    out = dir.stdout
    result = out.decode('utf-8')
    print("Results -- ", result)
    if 'T' in result:
        option = "S"
        return render(request,'Report.html',{"report": mark_safe(result),"uname" : u_name, "option":option})
    else:
        return render(request,'Report.html',{"report": mark_safe(result),"uname" : u_name})



def Help(request):
    if (u_name is None):
        return render(request,'Help.html',{"uname": U_name})
    else:
        option = "U"
        return render(request,'Help.html',{"uname": u_name, "option":option})
        

def FAQ(request):
    if (u_name is None):
        return render(request,'FAQ.html',{"uname": U_name})
    else:
        option = "U"
        return render(request,'FAQ.html',{"uname": u_name, "option":option})

 


   