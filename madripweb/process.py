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
import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black, blue
from reportlab.platypus import Image as img
from reportlab.pdfgen.canvas import Canvas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(BASE_DIR, 'madripweb/'))
import resnet50


image_name=sys.argv[1]
class Data_Preprocess:
    #Image name is passed on by the upload function of the user class
    
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
      
      os.chdir(os.path.join(BASE_DIR, 'madripweb/scans'))
      res=self.ResizeImage()
      os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/temp'))
      cv2.imwrite(self.ImageName,res)
      back = self.BackgroundReduction()
      cv2.imwrite(self.ImageName,back)
      cr=self.Crop()
      os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/Results'))
      cr.save(self.ImageName)
      ed=self.EdgeDetection()
      os.chdir(os.path.join(BASE_DIR, 'madripweb/static/BloodVImage'))
      plt.imsave(self.ImageName, ed, cmap='gray', format='jpeg')

      #remove the temp folder

      os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/temp'))
      os.remove(image_name)
      os.chdir(os.path.join(BASE_DIR, 'madripweb/scans/'))
      os.rmdir('temp')
      os.mkdir('temp')
      return True

#--------------Identification---------------

class testIMDataset(Dataset):
    """Face Landmarks dataset."""

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

class identifyDataset(Dataset):

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
        model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=False, init_weights=True)
        PATH = os.path.join(BASE_DIR, 'madripweb/dr3.pth')
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

        #sshow
        #path = os.path.join(BASE_DIR, 'madripweb\\static\\Results' + ftmap)
        self.resultantImage = ftmap
        path = '\\static\\ExudateImage\\' + ftmap
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\ExudateImage'))
        cv2.imwrite(ftmap, cv2.drawContours(imS, cnt, -1, (240, 120, 0), 2))
        return path

    def DisplayResult(self):
        path = self.Extract()
        print (path)

#---DME identification----------


class DME_Identification:
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
        
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\scans'))
        path = name
        imageTra = identifyDataset(imageName=path,transform=tra_test)
        img_loader = torch.utils.data.DataLoader(
            imageTra,
            batch_size=  1, shuffle=False,
            num_workers=  0, pin_memory=True)
    
        model = resnet50.resnet50(num_classes=  2, multitask=  True, liu=  False,
                chen=  False, CAN_TS=  False, crossCBAM=  True,
                crosspatialCBAM =   False,  choice=  False)
    
        model_dict = model.state_dict()
        pretrain_path = {"resnet50": "F:/madripweb/madripweb/resnet50-19c8e357.pth",}['resnet50']
        pretrained_dict = torch.load(pretrain_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict.pop('classifier.weight', None)
        pretrained_dict.pop('classifier.bias', None)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        CUDA_VISIBLE_DEVICES=""
        # torch.cuda.set_device( gpu )
        # model = model.cuda(  gpu )
        # optimizer = torch.optim.Adam(model.parameters(),   self.base_lr, weight_decay=  self.weight_decay)
        checkpoint = torch.load('F:/madripweb/madripweb/CANet(2).pth',map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optim_dict'])
        #print("model loaded")
        model.eval()
        model.to('cpu')
        with torch.no_grad():

            dataiter = iter(img_loader)
            input= dataiter.next()
            output = model(input)
            # torch.cuda.synchronize()
            output0 = output[0] #dr
            output1 = output[1] #dme
            output0 = torch.softmax(output0, dim=1)
            output1 = torch.softmax(output1, dim=1)
            all_output.append(output0.cpu().data.numpy())
            all_output_dme.append(output1.cpu().data.numpy())
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
            print("DME stage: ",dme)
        if option == "R":
            print("DR stage: ",dr)

#---Report generation-------------     

class Generate_Report:
    def __init__(self,sImage,eImage="",bImage="",rStage="1",mStage="1"):
        self.subjectImageName = sImage
        self.ExudateImageName = eImage
        self.BloodVImageName = bImage
        self.DR_stage = rStage
        self.DME_stage = mStage  

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

    def writeResult(self):
        os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\'))
        file = Canvas("Report.pdf", pagesize=A4)
        
            # os.chdir(os.path.join(BASE_DIR, 'madripweb\\static\\BloodVImage'))
            # img = Image(self.BloodVImageName)
        file.setLineWidth(.3)
        file.setTitle("Report")
        file.setFont('Helvetica', 12)
        file.drawString(30,750,'Diabetic Retinopathy Stage: ')
        file.drawString(200,750,self.DR_stage)
        file.drawString(500,750,"18/12/2020")
        #file.line(480,747,580,747)
        file.drawString(30,725,"Diabetic Macular Edema Stage: ")
        file.drawString(200,725,self.DME_stage)
        #file.line(378,723,580,723)
        file.drawString(400,703,'Report Generated By: ')
        #file.line(120,700,580,700)
        file.drawString(550,703,"MADRIP")
        file.save()

    def DisplayReport(self):
        # self.collectResult()
        self.writeResult()
        


#main


if sys.argv[2] == "P":
    obj = Data_Preprocess(image_name)
    check = obj.Preprocess_Upload()
    

if sys.argv[2] == "R":
    identify = DME_Identification()
    identify.DisplayDMEresult(image_name,sys.argv[2])
    
    
    

if sys.argv[2] == "M":
    identify = DME_Identification()
    identify.DisplayDMEresult(image_name,sys.argv[2])
    
    
    
    
if sys.argv[2] == "E":
    extract = Feature_Extraction(image_name)
    extract.DisplayResult()
    
    

if sys.argv[2] == "G":
    report = Generate_Report(image_name,image_name)
    report.DisplayReport()
    

