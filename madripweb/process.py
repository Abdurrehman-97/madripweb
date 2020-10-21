import cv2
import matplotlib.pyplot as plt
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
# #         _srcf.save(self.ImageName)
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
        # return None
    
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

      os.chdir('C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\scans')
      res=self.ResizeImage()
      os.chdir('C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\scans\\temp')
      cv2.imwrite(self.ImageName,res)
      back = self.BackgroundReduction()
      cv2.imwrite(self.ImageName,back)
      cr=self.Crop()
      os.chdir('C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\scans\\Results')
      cr.save(self.ImageName)
      # ed=self.EdgeDetection()
      # %cd "/content/drive/My Drive/User_Image/Result"
      # plt.imsave(self.ImageName, ed, cmap='gray', format='jpeg')

      #remove the temp folder

      os.chdir('C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\scans\\temp')
      os.remove(image_name)
      os.chdir('C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\scans')
      os.rmdir('temp')
      os.mkdir('temp')

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
        # model.to(device)
        dse = testIMDataset(imgnamelist,rootdir,trans)
        dseloader = torch.utils.data.DataLoader(dse, batch_size=1,
                                            shuffle=False, num_workers=0)
        dataiter = iter(dseloader)
        image= dataiter.next()
        output = model(image)
        _, predicted = torch.max(output.data,1)
        return(predicted.detach().numpy()[0])

    def DisplayResult(self):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=False)
        PATH = 'C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\dr3.pth'
        model.load_state_dict(torch.load(PATH))

        model.eval()
        model.to("cpu")

        t = transforms.Compose([transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        rootdir="C:\\Users\\DELL\\Desktop\\MADRIP_Web Module\\MADRIP_Web Module\\madripweb\\madripweb\\scans\\Results"
        imgnamelist=self.subjectImage
        print("Retinopathy Stage: ",Stage_Identification.detectStage(model,[imgnamelist],rootdir,t))

#main

obj = Data_Preprocess(image_name)
obj.Preprocess_Upload()
identify = Stage_Identification()
identify.setsubjectImage(image_name)
identify.DisplayResult()