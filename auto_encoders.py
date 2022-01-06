import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import time
# from skimage.io import imsave
import torch.nn.functional as f
from torch.autograd import Variable
class AE_3D_Net_R(torch.nn.Module):
  def __init__(self):
    super(AE_3D_Net_R, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv3d(1,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool3d((1,2,2),stride=(1,2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv3d(64,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv3d(128,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(256),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool3=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv4a=torch.nn.Sequential(
         torch.nn.Conv3d(256,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(512),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool4=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv5a=torch.nn.Sequential(
         torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(512),
         torch.nn.ReLU())#1*2*2
######----layer 
    self.Dec_Conv5a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(512),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool4 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv4a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(512,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(256),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool3 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(256,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(128,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(64,1,(3,3,3),stride=(1,1,1),padding=(1,1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      start=time.time()
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      conv_dec3_out = self.Dec_Conv3a(conv_dec4_out_pool)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out
      # print(out)
      # exit()
      return out

class AE_3D_Net_6(torch.nn.Module):
  def __init__(self):
    super(AE_3D_Net_6, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv3d(1,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool3d((1,2,2),stride=(1,2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv3d(64,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv3d(128,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(256),
         torch.nn.ReLU())#1*2*2

######----layer 

    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(256,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(128,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(64,1,(3,3,3),stride=(1,1,1),padding=(1,1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      start=time.time()
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)

      conv_dec3_out = self.Dec_Conv3a(conv_enc3_out)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out
      # print(out)
      # exit()
      return out

class AE_3D_woconnection_Net(torch.nn.Module):
  def __init__(self):
    super(AE_3D_woconnection_Net, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv3d(1,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool3d((1,2,2),stride=(1,2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv3d(64,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv3d(128,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(256),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool3=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv4a=torch.nn.Sequential(
         torch.nn.Conv3d(256,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(512),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool4=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv5a=torch.nn.Sequential(
         torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(512),
         torch.nn.ReLU())#1*2*2
######----layer 
    self.Dec_Conv5a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(512),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool4 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv4a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(512,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(256),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool3 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(256,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(128,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(64,1,(3,3,3),stride=(1,1,1),padding=(1,1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      start=time.time()
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool,ind1_ = self.Enc_pool1(conv_enc1_out)
      _,ind1 = self.Enc_pool1(torch.zeros(conv_enc1_out.size()).cuda())
      # print(ind1_.size())
      # print(ind1.size())
      # exit()
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      
      conv_enc2_out_pool,_ = self.Enc_pool2(conv_enc2_out)
      _,ind2 = self.Enc_pool2(torch.ones(conv_enc2_out.size()).cuda())
      
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      
      conv_enc3_out_pool,_= self.Enc_pool3(conv_enc3_out)

      _,ind3 = self.Enc_pool3(torch.ones(conv_enc3_out.size()).cuda())

      conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      
      conv_enc4_out_pool,ind4_= self.Enc_pool4(conv_enc4_out)
      _,ind4 = self.Enc_pool4(torch.ones(conv_enc4_out.size()).cuda())

      conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)
        
      conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out,ind4)
      conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out,ind3)
      conv_dec3_out = self.Dec_Conv3a(conv_dec4_out_pool)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out,ind2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out,ind1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out
      # print(out)
      # exit()
      return out
class AE_2D_Net(torch.nn.Module):
  def __init__(self):
    super(AE_2D_Net, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv2d(16,32,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(32),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv2d(32,64,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv2d(64,128,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv4a=torch.nn.Sequential(
         torch.nn.Conv2d(128,256,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(256),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv5a=torch.nn.Sequential(
         torch.nn.Conv2d(256,512,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(512),
         torch.nn.ReLU())#1*2*2
######----layer 
    self.Dec_Conv5a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(512,256,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(256),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv4a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(256,128,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(128,64,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(64,32,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(32),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(32,16,(3,3),stride=(1,1),padding=(1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      x=x.squeeze(1)
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      conv_dec3_out = self.Dec_Conv3a(conv_dec4_out_pool)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out.unsqueeze(1)
      # print(out)
      # exit()
      return out


class AE_2D_Net_3layers_2(torch.nn.Module):
  def __init__(self):
    super(AE_2D_Net_3layers_2, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv2d(16,128,(11,11),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv2d(128,64,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv2d(64,32,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(32),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    # self.Enc_Conv4a=torch.nn.Sequential(
         # torch.nn.Conv2d(128,256,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    # self.Enc_Conv5a=torch.nn.Sequential(
         # torch.nn.Conv2d(256,512,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(512),
         # torch.nn.ReLU())#1*2*2
######----layer 
    # self.Dec_Conv5a=torch.nn.Sequential(
         # torch.nn.ConvTranspose2d(512,256,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv4a=torch.nn.Sequential(
         # torch.nn.ConvTranspose2d(256,128,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(128),
         # torch.nn.ReLU())#1*2*2
    self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(32,64,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(64,128,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(128,16,(11,11),stride=(1,1),padding=(1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      x=x.squeeze(1)
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      # conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      # conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      # conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      # conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      # conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      # conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      # conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      # conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      conv_dec3_out = self.Dec_Conv3a(conv_enc3_out)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out.unsqueeze(1)
      # print(out)
      # exit()
      return out,conv_enc3_out

class AE_2D_Net_3layers_3(torch.nn.Module):
  def __init__(self):
    super(AE_2D_Net_3layers_3, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv2d(3,128,(11,11),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv2d(128,32,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(32),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv2d(32,8,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(8),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    # self.Enc_Conv4a=torch.nn.Sequential(
         # torch.nn.Conv2d(128,256,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    # self.Enc_Conv5a=torch.nn.Sequential(
         # torch.nn.Conv2d(256,512,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(512),
         # torch.nn.ReLU())#1*2*2
######----layer 
    # self.Dec_Conv5a=torch.nn.Sequential(
         # torch.nn.ConvTranspose2d(512,256,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv4a=torch.nn.Sequential(
         # torch.nn.ConvTranspose2d(256,128,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(128),
         # torch.nn.ReLU())#1*2*2
    self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(8,32,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(32),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(32,128,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(128,3,(11,11),stride=(1,1),padding=(1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      x=x.squeeze(1)
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      # conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      # conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      # conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      # conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      # conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      # conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      # conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      # conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      conv_dec3_out = self.Dec_Conv3a(conv_enc3_out)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out.unsqueeze(1)
      # print(out)
      # exit()
      return out,conv_enc3_out
class AE_2D_Net_3layers(torch.nn.Module):
  def __init__(self):
    super(AE_2D_Net_3layers, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv2d(16,512,(11,11),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(512),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv2d(512,256,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(256),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv2d(256,128,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    # self.Enc_Conv4a=torch.nn.Sequential(
         # torch.nn.Conv2d(128,256,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    # self.Enc_Conv5a=torch.nn.Sequential(
         # torch.nn.Conv2d(256,512,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(512),
         # torch.nn.ReLU())#1*2*2
######----layer 
    # self.Dec_Conv5a=torch.nn.Sequential(
         # torch.nn.ConvTranspose2d(512,256,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv4a=torch.nn.Sequential(
         # torch.nn.ConvTranspose2d(256,128,(3,3),stride=(1,1),padding=(1,1)),
         # torch.nn.BatchNorm2d(128),
         # torch.nn.ReLU())#1*2*2
    self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv3a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(128,256,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(256),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool2 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(256,512,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(512),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(512,16,(11,11),stride=(1,1),padding=(1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      x=x.squeeze(1)
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      # conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      # conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      # conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      # conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      # conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      # conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      # conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      # conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      conv_dec3_out = self.Dec_Conv3a(conv_enc3_out)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      # out = conv_dec1_out.unsqueeze(1)
      # print(out)
      # exit()
      return out,conv_enc3_out

KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2
class ConvLSTMCell(torch.nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        # print(input_.size()[0])
        # exit()
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size).cuda()),
                Variable(torch.zeros(state_size).cuda())
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell

class ConvLSTM_Net(torch.nn.Module):
  def __init__(self):
    super(ConvLSTM_Net, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv3d(1,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool3d((1,2,2),stride=(1,2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv3d(64,128,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(128),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    
    self.Enc_Conv3a=ConvLSTMCell(128,256)#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv4a=ConvLSTMCell(256,512)#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv5a=ConvLSTMCell(512,512)#1*2*2
######----layer 
    self.Dec_Conv5a=ConvLSTMCell(512,512)#1*2*2
    # self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv4a=ConvLSTMCell(512,256)#1*2*2
    # self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    self.Dec_Conv3a=ConvLSTMCell(256,128)#1*2*2


    self.Dec_unpool2 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(128,64,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         torch.nn.BatchNorm3d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose3d(64,1,(3,3,3),stride=(1,1,1),padding=(1,1,1)))

    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      # start=time.time()
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool, self.indices1 = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool, self.indices2 = self.Enc_pool2(conv_enc2_out)
      conv_enc2_out_pool_=conv_enc2_out_pool.permute(2,0,1,3,4)
      conv_dec3_out=torch.zeros(conv_enc2_out_pool_.size()).cuda()
      # print(conv_enc2_out_pool_.size())
      # exit()
      state_1 = None
      state_2 = None
      state_3 = None
      state_4 = None
      state_5 = None
      state_6 = None
      for t in range(0, 8):
          state_1 = self.Enc_Conv3a(conv_enc2_out_pool_[t], state_1)
          state_2 = self.Enc_Conv4a(state_1[0], state_2)
          state_3 = self.Enc_Conv5a(state_2[0], state_3)
          state_4 = self.Dec_Conv5a(state_3[0], state_4)
          state_5 = self.Dec_Conv4a(state_4[0], state_5)
          state_6 = self.Dec_Conv3a(state_5[0], state_6)
          conv_dec3_out[t]=state_6[0]
      # conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool_)
      # conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      # conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      # conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      # conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      # conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      # conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      # conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      # conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      # conv_dec3_out = self.Dec_Conv3a(conv_dec4_out_pool)
      conv_dec3_out_=conv_dec3_out.permute(1,2,0,3,4)
      conv_dec3_out_pool = self.Dec_unpool2(conv_dec3_out_, self.indices2)
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_pool)
      conv_dec2_out_pool = self.Dec_unpool1(conv_dec2_out, self.indices1)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out_pool)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = conv_dec1_out
      # print(out)
      # exit()
      return out

class ConvLSTM_Net_3layers3x3(torch.nn.Module):
  def __init__(self):
    super(ConvLSTM_Net_3layers3x3, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv2d(1,64,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#9*56*56
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv2d(64,32,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(32),
         torch.nn.ReLU())#3*27*27
    
    self.Enc_Conv3a=ConvLSTMCell(32,32)#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv4a=ConvLSTMCell(32,32)#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv5a=ConvLSTMCell(32,32)#1*2*2
######----layer 
    # self.Dec_Conv5a=ConvLSTMCell(512,512)#1*2*2
    # self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv4a=ConvLSTMCell(512,256)#1*2*2
    # self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv3a=ConvLSTMCell(256,128)#1*2*2


    # self.Dec_unpool2 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(32,64,(3,3),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#1*2*2
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(64,1,(3,3),stride=(1,1),padding=(1,1)))
    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      #x:b,c,t,w,h
      batch=x.size()[0]
      tem=x.size()[2]
      x_=x.permute(0,2,1,3,4)
      x_=torch.reshape(x_,(batch*tem,x_.size()[2],x_.size()[3],x_.size()[4]))
      #x_:b*t,c,w,h
      # x=x.squeeze(1)
      conv_enc1_out = self.Enc_Conv1a(x_)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out)
      # print(conv_enc2_out_pool_.size())
      # exit()
      state_1 = None
      state_2 = None
      state_3 = None
      conv_enc2_out_=torch.reshape(conv_enc2_out,(tem,batch,conv_enc2_out.size()[1],conv_enc2_out.size()[2],conv_enc2_out.size()[3]))
      conv_dec3_out=torch.zeros(conv_enc2_out_.size()).cuda()
      # batch, channel, W, H
      for t in range(0, tem):
          state_1 = self.Enc_Conv3a(conv_enc2_out_[t], state_1)
          state_2 = self.Enc_Conv4a(state_1[0], state_2)
          state_3 = self.Enc_Conv5a(state_2[0], state_3)
          conv_dec3_out[t]=state_3[0]
      # conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool_)
      # conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      # conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      # conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      # conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      # conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      # conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      # conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      # conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      # conv_dec3_out = self.Dec_Conv3a(conv_dec4_out_pool)
      # conv_dec3_out_=conv_dec3_out.permute(1,2,0,3,4)
      conv_dec3_out_=torch.reshape(conv_dec3_out,(batch*tem,conv_dec3_out.size()[2],conv_dec3_out.size()[3],conv_dec3_out.size()[4]))
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = torch.reshape(conv_dec1_out,(batch,tem,conv_dec1_out.size()[1],conv_dec1_out.size()[2],conv_dec1_out.size()[3]))
      out_=out.permute(0,2,1,3,4)
      # print(out)
      # exit()
      return out_,conv_dec2_out

class ConvLSTM_Net_3layers(torch.nn.Module):
  def __init__(self):
    super(ConvLSTM_Net_3layers, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv2d(1,128,(11,11),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#9*56*56
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv2d(128,64,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(64),
         torch.nn.ReLU())#3*27*27
    
    self.Enc_Conv3a=ConvLSTMCell(64,32)#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool3=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv4a=ConvLSTMCell(32,32)#1*2*2
    # self.Enc_Conv4b=torch.nn.Sequential(
         # torch.nn.Conv3d(512,512,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())#1*2*2
    # self.Enc_pool4=torch.nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)#3*5*5
    self.Enc_Conv5a=ConvLSTMCell(32,64)#1*2*2
######----layer 
    # self.Dec_Conv5a=ConvLSTMCell(512,512)#1*2*2
    # self.Dec_unpool4 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv4a=ConvLSTMCell(512,256)#1*2*2
    # self.Dec_unpool3 = torch.nn.MaxUnpool2d((2,2), stride=(2,2))
    # self.Dec_Conv3a=ConvLSTMCell(256,128)#1*2*2


    # self.Dec_unpool2 = torch.nn.MaxUnpool3d((2,2,2), stride=(2,2,2))
    self.Dec_Conv2a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(64,128,(5,5),stride=(1,1),padding=(1,1)),
         torch.nn.BatchNorm2d(128),
         torch.nn.ReLU())#1*2*2
    self.Dec_Conv1a=torch.nn.Sequential(
         torch.nn.ConvTranspose2d(128,1,(11,11),stride=(1,1),padding=(1,1)))
    # self.conv_dec1=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(128,256,(3,3,3),stride=(2,2,2)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())
    # self.unpool1 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec2=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(256,512,(3,5,5),stride=(3,3,3),padding=(0,2,2)),
         # torch.nn.BatchNorm3d(512),
         # torch.nn.ReLU())
    # self.unpool2 = torch.nn.MaxUnpool3d((1,2,2), stride=(1,2,2))
    # self.conv_dec3=torch.nn.Sequential(
         # torch.nn.ConvTranspose3d(512,1,(2,10,10),stride=(1,4,4),padding=(0,3,3)))

  def forward(self, x):
      #x:b,c,t,w,h
      batch=x.size()[0]
      tem=x.size()[2]
      x_=x.permute(0,2,1,3,4)
      x_=torch.reshape(x_,(batch*tem,x_.size()[2],x_.size()[3],x_.size()[4]))
      #x_:b*t,c,w,h
      # x=x.squeeze(1)
      conv_enc1_out = self.Enc_Conv1a(x_)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out)
      # print(conv_enc2_out_pool_.size())
      # exit()
      state_1 = None
      state_2 = None
      state_3 = None
      conv_enc2_out_=torch.reshape(conv_enc2_out,(tem,batch,conv_enc2_out.size()[1],conv_enc2_out.size()[2],conv_enc2_out.size()[3]))
      conv_dec3_out=torch.zeros(conv_enc2_out_.size()).cuda()
      # batch, channel, W, H
      for t in range(0, tem):
          state_1 = self.Enc_Conv3a(conv_enc2_out_[t], state_1)
          state_2 = self.Enc_Conv4a(state_1[0], state_2)
          state_3 = self.Enc_Conv5a(state_2[0], state_3)
          conv_dec3_out[t]=state_3[0]
      # conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool_)
      # conv_enc3_out_pool, self.indices3 = self.Enc_pool3(conv_enc3_out)
      # conv_enc4_out = self.Enc_Conv4a(conv_enc3_out_pool)
      # conv_enc4_out_pool, self.indices4 = self.Enc_pool4(conv_enc4_out)
      # conv_enc5_out = self.Enc_Conv5a(conv_enc4_out_pool)

      # conv_dec5_out = self.Dec_Conv5a(conv_enc5_out)
      # conv_dec5_out_pool = self.Dec_unpool4(conv_dec5_out, self.indices4)
      # conv_dec4_out = self.Dec_Conv4a(conv_dec5_out_pool)
      # conv_dec4_out_pool = self.Dec_unpool3(conv_dec4_out, self.indices3)
      # conv_dec3_out = self.Dec_Conv3a(conv_dec4_out_pool)
      # conv_dec3_out_=conv_dec3_out.permute(1,2,0,3,4)
      conv_dec3_out_=torch.reshape(conv_dec3_out,(batch*tem,conv_dec3_out.size()[2],conv_dec3_out.size()[3],conv_dec3_out.size()[4]))
      conv_dec2_out = self.Dec_Conv2a(conv_dec3_out_)
      conv_dec1_out = self.Dec_Conv1a(conv_dec2_out)
      
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      out = torch.reshape(conv_dec1_out,(batch,tem,conv_dec1_out.size()[1],conv_dec1_out.size()[2],conv_dec1_out.size()[3]))
      out_=out.permute(0,2,1,3,4)
      # print(out)
      # exit()
      return out_,conv_dec2_out
class Discrimater_3D(torch.nn.Module):
  def __init__(self):
    super(Discrimater_3D, self).__init__()
    self.Enc_Conv1a=torch.nn.Sequential(
         torch.nn.Conv3d(1,2,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(8),
         torch.nn.ReLU())#9*56*56
    self.Enc_pool1=torch.nn.MaxPool3d((1,2,2),stride=(1,2,2),return_indices=True)#9*28*28
    self.Enc_Conv2a=torch.nn.Sequential(
         torch.nn.Conv3d(2,4,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(16),
         torch.nn.ReLU())#3*10*10
    self.Enc_pool2=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.Enc_Conv3a=torch.nn.Sequential(
         torch.nn.Conv3d(4,8,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(32),
         torch.nn.ReLU())#1*2*2
    # self.Enc_Conv3b=torch.nn.Sequential(
         # torch.nn.Conv3d(256,256,(3,3,3),stride=(1,1,1),padding=(1,1,1)),
         # torch.nn.BatchNorm3d(256),
         # torch.nn.ReLU())#1*2*2
    self.Enc_pool3=torch.nn.MaxPool3d((2,2,2),stride=(2,2,2),return_indices=True)#3*5*5
    self.classifier = torch.nn.Sequential(
            torch.nn.Linear(8*4*28*28, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(inplace=False),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid())


  def forward(self, x):
      start=time.time()
      conv_enc1_out = self.Enc_Conv1a(x)
      conv_enc1_out_pool,ind1_ = self.Enc_pool1(conv_enc1_out)
      conv_enc2_out = self.Enc_Conv2a(conv_enc1_out_pool)
      conv_enc2_out_pool,_ = self.Enc_pool2(conv_enc2_out)
      conv_enc3_out = self.Enc_Conv3a(conv_enc2_out_pool)
      conv_enc3_out_pool,_= self.Enc_pool3(conv_enc3_out)
      conv_pool = conv_enc3_out_pool.view(conv_enc3_out_pool.size(0), -1)
      out=self.classifier(conv_pool)
      # print(out)
      # exit()
      
      # conv_dec4_out = self.conv_dec2(conv_dec5_out_pool)
      # conv_dec2_out_pool = self.unpool2(conv_dec2_out, self.indices1)
      # conv_dec3_out = self.conv_dec3(conv_dec2_out_pool)
      # out = conv_pool
      # print(out)
      # exit()
      return out
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        # super解决在多重继承中调用父类可能出现的问题
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  # 全连接层输入的是一维向量，第一层节点个数120是根据Pytorch官网demo设定
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10因为使用的是cifar10，分为10类

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input (3,32,32)  output(16, 32-5+1=28, 32-5+1)
        x = self.pool1(x)  # output(16, 28/2=14, 28/2)
        x = F.relu((self.conv2(x)))  # output(32, 14-5+1=10, 14-5+1=10)
        x = self.pool2(x)  # output(32, 10/2=5, 10/2=5)
        x = x.view(-1, 32*5*5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = F.relu(self.fc3(x))  # output(10)
        return x
class VGG(nn.Module):
    """
    VGG builder
    """
    def __init__(self, arch: object, num_classes=10) -> object:
        super(VGG, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return self.fc3(out)
def VGG_11(n):
    return VGG([1, 1, 2, 2, 2], num_classes=n)

def VGG_13():
    return VGG([1, 1, 2, 2, 2], num_classes=n)

def VGG_16():
    return VGG([2, 2, 3, 3, 3], num_classes=n)

def VGG_19():
    return VGG([2, 2, 4, 4, 4], num_classes=n)