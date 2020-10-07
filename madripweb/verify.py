import pymongo
from pymongo import MongoClient
import sys

class User:
    def __init__(self):
        
        self.Name=""
        self.userName=""
        self.Email=""
        self.Phn=0
        self.Organisation=""
        self.Password=""
        self.fileName=""
        pass
        
    def setName(self,name):
        self.Name=name
        pass
    
    def setuserName(self,uname):
        self.userName=uname
        pass
    
    def setPassword(self,pwrd):
        self.Password=pwrd
        pass
        
    def setEmail(self,email):
        self.Email=email
        pass
        
    def setPhn(self,phn):
        self.Phn=phn
        pass
        
    def setOrganisation(self,org):
        self.Organisation=org
        pass

        
    def getName(self):
        return self.Name
    
    def getuserName(self):
        return self.userName
    
    def getPassword(self):
        return self.Password
    
    def getEmail(self):
        return self.Email
    
    def getPhone(self):
        return self.Phn
    
    def getOrganisation(self):
        return self.Organisation

    def getfileName(self):
      return self.fileName
    
    def SignIn(self,uname,password):
        #verrify the user details
        result = collection.find({'UserName':uname})
        for i in result:
          if i["Password"] == password:
            return True
          else:
            return False
        
    
    def Register(self):
        #create a new user
        post = {"Name": self.Name, "Email": self.Email,"Organisation" : self.Organisation,"Phone" : self.Phn, "UserName" : self.userName, "Password" : self.Password}
        if collection.insert_one(post):
             #user created and stored
             return True
        else:
          return False

    
    def UploadImage(self):
        #this function will run behind the upload button
        
        # %cd "/content/drive/My Drive/User_Image"
        # uploaded=files.upload()
        # for key, value in uploaded.items():
        #       self.fileName = key
        pass


#Main----------------------

cluster = MongoClient('mongodb://test_user:testuser123@fyp-shard-00-00-djj0y.gcp.mongodb.net:27017,fyp-shard-00-01-djj0y.gcp.mongodb.net:27017,fyp-shard-00-02-djj0y.gcp.mongodb.net:27017/test?ssl=true&replicaSet=FYP-shard-0&authSource=admin&retryWrites=true&w=majority')
db = cluster["MADRIP"]
collection = db["Users"]
u = User()

if sys.argv[1] == "R":
    
    u.setName(sys.argv[2]) 
    u.setEmail( sys.argv[3])
    u.setOrganisation(sys.argv[4]) 
    u.setPhn(sys.argv[5]) 
    u.setuserName(sys.argv[6])
    u.setPassword(sys.argv[7])
    check=u.Register()
    if check:
        print("Successful registration")
    else:
        print("Registration error")

if sys.argv[1] == "L":
    
    check=u.SignIn(sys.argv[2],sys.argv[3])

    if check:
        print("Log In successful")
    else:
        print("Cannot Log In")
