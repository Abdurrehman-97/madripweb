#contains code for pre-processing,identification,extraction and report generation...

import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd
import os
import PIL
from PIL import Image, ImageDraw, ImageFilter
import csv
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage
#reportlab
import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black, blue
from reportlab.platypus import Image as img
from reportlab.pdfgen.canvas import Canvas
import time
import PyPDF2   
from PyPDF2 import PdfFileWriter, PdfFileReader
import io 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(BASE_DIR, 'madripweb/'))
import resnet50
#pdfminer 
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
import pdfminer


#retrieving image name passed on by the function in views.py...
image_name=sys.argv[1]   

# Data-Preprocessing............
class Data_Preprocess:
    #Image name is passed as a an argumment to the constructor..
    
    def __init__(self,iname):
        
        self.ImageName=iname
        pass
    
    def setImageName(self,iname):
        self.ImageName=iname
        pass
    
    def getImageName(self):
        return self.ImageName
    
    def ReadImage(self):
        return cv2.imread(self.ImageName)
    
    def ResizeImage(self):
#         res = Image.open(self.ImageName)
#         basewidth = 100
#         wpercent = (basewidth / float(res.size[0]))
#         hsize = int((float(res.size[1]) * float(wpercent)))
#         res = res.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
#         _srcf.save(self.ImageName)
        res = cv2.imread(self.ImageName,cv2.IMREAD_UNCHANGED)
        scale_percent = 10 # percent of the original image size
        width = int(res.shape[1] * scale_percent / 100)
        height = int(res.shape[0] * scale_percent / 100)
        dimension = (width, height)
        resized = cv2.resize(res , dimension, interpolation = cv2.INTER_AREA)

        return resized
    
    def BackgroundReduction(self):
        br=cv2.imread(self.ImageName)
        gray = cv2.cvtColor(br,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
        
        # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        result = br[y:y+h,x:x+w]
        return result
        
    
    def Crop(self):

        
        background_color=255,255,255
        blur_radius=0
        offset=0
        
        crop_img=Image.open(self.ImageName)
        background = Image.new(crop_img.mode, crop_img.size, background_color)
        offset = blur_radius * 2 + offset
        mask = Image.new("L", crop_img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((offset, offset, crop_img.size[0]-1.01, crop_img.size[1] ), fill=255) #co-ordinates starting from the left top corner
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        if(mask != None):

          crop_img=Image.composite(crop_img, background,mask)
        
        return crop_img
        
          
    def EdgeDetection(self): #returns np array

        
        init_img=cv2.imread(self.ImageName) #initial image
        #blur_img = cv2.medianBlur(init_img, 5)
        img = np.array(init_img).astype(np.uint8)
        
        # Apply gray scale
        gray_img = np.round(0.299 * img[:, :, 0] +
                      0.587 * img[:, :, 1] +
                      0.114 * img[:, :, 2]).astype(np.uint8)
         
        # Prewitt Operator
        h, w = gray_img.shape
        # define filters
        horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
        vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1

        # define images with 0s
        newgradientImage = np.zeros((h, w))
        #gc.collect()
        
        # offset by 1
        for i in range(1, h - 1):
            
            for j in range(1, w - 1):
                
                horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                              (horizontal[0, 1] * gray_img[i - 1, j]) + \
                              (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                              (horizontal[1, 0] * gray_img[i, j - 1]) + \
                              (horizontal[1, 1] * gray_img[i, j]) + \
                              (horizontal[1, 2] * gray_img[i, j + 1]) + \
                              (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                              (horizontal[2, 1] * gray_img[i + 1, j]) + \
                              (horizontal[2, 2] * gray_img[i + 1, j + 1])

                verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                                (vertical[0, 1] * gray_img[i - 1, j]) + \
                                (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                                (vertical[1, 0] * gray_img[i, j - 1]) + \
                                (vertical[1, 1] * gray_img[i, j]) + \
                                (vertical[1, 2] * gray_img[i, j + 1]) + \
                                (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                                (vertical[2, 1] * gray_img[i + 1, j]) + \
                                (vertical[2, 2] * gray_img[i + 1, j + 1])

              # Edge Magnitude
                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                newgradientImage[i - 1, j - 1] = mag
        
        return newgradientImage
    


    def Preprocess_Upload(self):
        #changing working directory...to the directory where the scan is stored
        os.chdir(os.path.join(BASE_DIR, 'madripweb/scans'))
        res=self.ResizeImage()

        #changing the directory to store the result..
        os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/temp'))
        cv2.imwrite(self.ImageName,res)
        back = self.BackgroundReduction()
        cv2.imwrite(self.ImageName,back)
        cr=self.Crop()

        #saving the cropped image in Results directory for use for EdgeDetection..
        os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/Results'))
        cr.save(self.ImageName)
        ed=self.EdgeDetection()

        #saving image in static directory for displaying purpose..
        os.chdir(os.path.join(BASE_DIR, 'madripweb/static/BloodVImage'))
        plt.imsave(self.ImageName, ed, cmap='gray', format='jpeg')

        #remove and re-make the temp directory
        os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/temp'))
        os.remove(image_name)
        os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/'))
        os.rmdir('temp')
        os.mkdir('temp')
        return True

#--------------Identification---------------

class testIMDataset(Dataset):
    #for creating dataset for the subjective test image..
    def __init__(self, listNm, root_dir, transform=None):
        self.fundus_samples = listNm
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fundus_samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = Image.open(self.root_dir+"/"+self.fundus_samples[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class Stage_Identification:

    def __init__(self):
        self.Stage=-1
        self.stageDescription=""
        self.subjectImage=""

    def setStage(self,stage):
        self.Stage=stage
        pass
  
    def setstageDescription(self,desc):
        self.stageDescription=desc
        pass

    def setsubjectImage(self,name):
        self.subjectImage=name
        pass

    def getStage(self):
        return self.Stage

    def getstageDescription(self):
        return self.stageDescription

    def getsubjectImage(self):
        return self.subjectImage


    def detectStage(self,model,imgnamelist,rootdir,trans):
        model.eval()
        dse = testIMDataset(imgnamelist,rootdir,trans)
        dseloader = torch.utils.data.DataLoader(dse, batch_size=1,
                                            shuffle=False, num_workers=0)
        dataiter = iter(dseloader)
        image= dataiter.next()
        output = model(image)
        _, predicted = torch.max(output.data,1)
        return(predicted.detach().numpy()[0])

    def DisplayResult(self):
        #loading the base model from pytorch library..
        model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=False, init_weights=True)
        PATH = os.path.join(BASE_DIR, 'madripweb/dr3.pth')

        #loading the trained model dr3.pth
        model.load_state_dict(torch.load(PATH,map_location='cpu'))
        model.eval()
        model.to("cpu")

        t = transforms.Compose([transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        rootdir=os.path.join(BASE_DIR, 'madripweb\\scans\\Results')
        imgnamelist=self.subjectImage
        global stage
        stage = self.detectStage(model,[imgnamelist],rootdir,t)
        print("Retinopathy stage: ", stage)

#-----Feature extraction---------------
class Feature_Extraction:

    def __init__(self,image):

        self.subjectImage = image
        self.resultantImage = ""

    def Extract(self):
        #changing the working directory..to the directory whre scans are stored..
        os.chdir(os.path.join(BASE_DIR, 'madripweb/scans'))

        imgName = self.subjectImage
        
        img = cv2.imread(imgName)

        imS = cv2.resize(img, (512, 512))

        green_image = imS.copy()
        green_image[:, :, 0] = 0
        green_image[:, :, 2] = 0

        gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        newName = imgName.split('.')
        newName2 = deepcopy(newName)
        newName.insert(1, '_exudate.')
        exudateImg = ""
        for i in newName:
            exudateImg += i
        newName2.insert(1, '_ftmaped.')
        ftmap = ""
        for i in newName2:
            ftmap += i


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        img_dilation = cv2.dilate(thresh1, kernel, iterations=2)
        img_erosion = cv2.erode(img_dilation, kernel, iterations=2)

        # cv2.imshow('Input', thresh1)
        # cv2.imshow('Dilation', img_dilation)
        # cv2.imshow('Erosion', img_erosion)

        newName = imgName.split('.')
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\scans'))

        #saving the gray scale image of the extracted exudate..
        cv2.imwrite(exudateImg, img_erosion)

        _, cnt, _ = cv2.findContours(
            img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        def removearray(L, arr):
            ind = 0
            size = len(L)
            while ind != size and not np.array_equal(L[ind], arr):
                ind += 1
            if ind != size:
                L.pop(ind)
            else:
                raise ValueError('array not found in list.')

        i = 0
        while i < 2 and i < len(cnt):
            c = deepcopy(max(cnt, key=cv2.contourArea))
            removearray(cnt, c)
            i = i+1

        
        #path = os.path.join(BASE_DIR, 'madripweb\\static\\Results' + ftmap)
        self.resultantImage = ftmap
        
        path = '\\static\\ExudateImage\\' + ftmap

        #saving the final image; exudates mapped on scan..
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\ExudateImage'))
        cv2.imwrite(ftmap, cv2.drawContours(imS, cnt, -1, (240, 120, 0), 2))
        return path

    def DisplayResult(self):
        path = self.Extract()
        print (path)

#---DME identification----------

class identifyDataset(Dataset):
    #for dataset creation for the respective test image..
    def __init__(self,imageName,transform):
        self.imageName = imageName
        self.transform = transform

    def __len__(self):
        return len(self.imageName)

    def __getitem__(self, idx):
        img = Image.open(self.imageName)
        image = img.convert('RGB')
        image = self.transform(image)
        return image


class DME_Identification:
    # Code for  model training was the result of the following paper:
    # CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading
    
    def __init__(self):
        self.Stage=-1
        self.DRstage = -1
        self.stageName=""
        self.ImageName=""

    def identifyDME(self,name,gpu=None):

        self.ImageName = name
        all_output = []
        all_output_dme = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        tra_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

        #change directory where original scan is stored..
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\scans'))
        path = name

        #pass image to identifyDataset..
        imageTra = identifyDataset(imageName=path,transform=tra_test)

        #load the dataset..
        img_loader = torch.utils.data.DataLoader(
            imageTra,
            batch_size=  1, shuffle=False,
            num_workers=  0, pin_memory=True)

        #create resnet50 object..class resnet50 present in resnet50.py
        model = resnet50.resnet50(num_classes=  2, multitask=  True, liu=  False,
                chen=  False, CAN_TS=  False, crossCBAM=  True,
                crosspatialCBAM =   False,  choice=  False)
    
        model_dict = model.state_dict()

        #load the pretrained model resnet50..downloaded from pytorch.org/models
        pretrain_path = {"resnet50": "F:/madripweb/madripweb/resnet50-19c8e357.pth",}['resnet50']
        pretrained_dict = torch.load(pretrain_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict.pop('classifier.weight', None)
        pretrained_dict.pop('classifier.bias', None)

        # update & load the model dictionary
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        CUDA_VISIBLE_DEVICES=""
        
        # torch.cuda.set_device( gpu )
        # model = model.cuda(  gpu )
        
        #load the trained model & load its dictionary..
        checkpoint = torch.load('F:/madripweb/madripweb/CANet(2).pth',map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        #print("model loaded")
        model.eval()
        model.to('cpu')

        with torch.no_grad():

            #passing the image to the model..
            dataiter = iter(img_loader)
            input= dataiter.next()
            output = model(input)
            # torch.cuda.synchronize()
            output0 = output[0] #dr stage
            output1 = output[1] #dme stage
            output0 = torch.softmax(output0, dim=1)
            output1 = torch.softmax(output1, dim=1)
            all_output.append(output0.cpu().data.numpy()) #for dr
            all_output_dme.append(output1.cpu().data.numpy()) #for dme
            # returning value--parent loop--child loop
            all_output = [item for sublist in all_output for item in sublist]
            all_output_dme = [item for sublist in all_output_dme for item in sublist]
            pr_dr = np.argmax(all_output,axis=1)
            pr_dme = np.argmax(all_output_dme,axis=1)

            self.Stage = pr_dme[0]
            self.DRstage = pr_dr[0]
            return (pr_dr[0],pr_dme[0])
            # print(pr_dr[0],pr_dme[0])

    def DisplayDMEresult(self,name,option):

        dr , dme = self.identifyDME(name)
        if option == "M":
            print(dme)
        if option == "R":
            print(dr)

#---Report generation-------------     
class Generate_Report:
    def __init__(self):
        self.subjectImageName = ""
        self.ExudateImageName = ""
        self.BloodVImageName = ""
        self.DR_stage = ""
        self.DME_stage = ""

    def setsubjectImage(self,image):
        self.subjectImageName = image
    def setExudateImage(self,image):
        self.ExudateImageName = image
    def setBloodVImage(self,image):
        self.BloodVImageName = image
    def setDR_stage(self,stage):
        self.DR_stage = stage
    def setDME_stage(self,stage):
        self.DME_stage = stage
    
    def collectResult(self):
        # if obj != None and extract != None and identify != None:
        #     self.subjectImageName = obj.ImageName
        #     self.subjectImageName = extract.resultantImage
        #     self.BloodVImageName = obj.ImageName
        #     self.DR_stage = identify.DRstage
        #     self.DME_stage = identify.Stage
        #     return True
        # else:
        #     return False
        pass

    def Locating(self,lt_objs):
        # utility function for finding content location in pdf
        # loop over the object list
        for obj in lt_objs:

            # if it's a textbox, print text and location
            if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
                print(obj.bbox[0], obj.bbox[1], obj.get_text().replace('\n', '_'))

            # if it's a container, recurse
            elif isinstance(obj, pdfminer.layout.LTFigure):
                self.Locating(obj._objs)


    def writeResult(self):
        
        #Report name respective to patient ID..
        #create new pdf with results then merge it with the template..
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\'))
        name = "Report_" + sys.argv[7] + ".pdf"
        packet = io.BytesIO()
        # create a new PDF with Reportlab
        can = Canvas(packet, pagesize=A4)
        can.drawString(270,344.616157418202,sys.argv[3])
        can.drawString(270,305.07963365820206,sys.argv[4])
        can.save()

        #move to the beginning of the StringIO buffer
        packet.seek(0)
        new_pdf = PdfFileReader(packet)
        # read your existing PDF
        existing_pdf = PdfFileReader(open("Report.pdf", "rb"))
        output = PdfFileWriter()
        # add the the new pdf on the existing page
        page = existing_pdf.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)
        # finally, write "output" to a real file
        outputStream = open(name, "wb")
        output.write(outputStream)
        outputStream.close()

        #----------------------------------------------
        # file = Canvas(name, pagesize=A4)
        
        #     # os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\BloodVImage'))
        #     # img = Image(self.BloodVImageName)
        # file.line(0,0,0,600)
        # file.setLineWidth(.3)
        # file.setTitle("Report")
        # file.setFont('Helvetica', 12)
        # file.drawString(30,800,'Subject Image: ')
        # file.drawString(200,800,self.subjectImageName)
        # file.drawString(30,775,'Exudate Image: ')
        # file.drawString(200,775,sys.argv[5])
        # file.drawString(30,750,'Diabetic Retinopathy Stage: ')
        # file.drawString(200,750,sys.argv[3])
        # file.drawString(500,750,"18/12/2020")
        # #file.line(480,747,580,747)
        # file.drawString(30,725,"Diabetic Macular Edema Stage: ")
        # file.drawString(200,725,sys.argv[4])
        # #file.line(378,723,580,723)
        # file.drawString(400,703,'Report Generated By: ')
        # #file.line(120,700,580,700)
        # file.drawString(530,703,"MADRIP")
        # file.save()

        # Open a PDF file to find content locations-------------
        # source: stackoverflow
        # os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\'))
        # fp = open('Report.pdf', 'rb')

        # # Create a PDF parser object associated with the file object.
        # parser = PDFParser(fp)

        # # Create a PDF document object that stores the document structure.
        # # Password for initialization as 2nd parameter
        # document = PDFDocument(parser)

        # # Check if the document allows text extraction. If not, abort.
        # if not document.is_extractable:
        #     raise PDFTextExtractionNotAllowed

        # # Create a PDF resource manager object that stores shared resources.
        # rsrcmgr = PDFResourceManager()

        # # Create a PDF device object.
        # device = PDFDevice(rsrcmgr)

        # # Layout Analysis
        # # Set parameters for analysis.
        # laparams = LAParams()

        # # Create a PDF page aggregator object.
        # device = PDFPageAggregator(rsrcmgr, laparams=laparams)

        # # Create a PDF interpreter object.
        # interpreter = PDFPageInterpreter(rsrcmgr, device)

        # # loop over all pages in the document
        # for page in PDFPage.create_pages(document):

        #     # read the page into a layout object
        #     interpreter.process_page(page)
        #     layout = device.get_result()

        #     # extract text from this object
        #     self.Locating(layout._objs)
        
        return True

    def DisplayReport(self):
        # self.collectResult()
        if self.writeResult():
            return True
        else:
            return False

  

#Main...........
# calling the respective functions 
# according to the option passed on by the calling function in views.py..
if sys.argv[2] == "P":
    obj = Data_Preprocess(image_name)
    check = obj.Preprocess_Upload()
    if check:
        print("True")
    else:
        print("False")
    
if sys.argv[2] == "R":
    identify =DME_Identification()
    identify.DisplayDMEresult(image_name,sys.argv[2])

    # Following code runs the dr3.pth model
    # identify =Stage_Identification()
    # identify.setsubjectImage(image_name)
    # identify.DisplayResult()
        
if sys.argv[2] == "M":
    identify = DME_Identification()
    identify.DisplayDMEresult(image_name,sys.argv[2])
            
if sys.argv[2] == "E":
    extract = Feature_Extraction(image_name)
    extract.DisplayResult()  
    
if sys.argv[2] == "G":
    report = Generate_Report()
    report.setsubjectImage(image_name) 
    
    if report.DisplayReport():
        print("True")
    else:
        print("False")
