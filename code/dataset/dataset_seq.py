# -*- coding:utf-8 -*-
import os
# import cv2
from PIL import Image
from torch.utils import data
import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import transforms as T


class Arotacls_seq(data.Dataset):
    
    def __init__(self,root,transforms=None,mode=None):
        '''
        Get images, divide into train/val set
        '''
        self.mode = mode
        self.data_root = root
        self.transforms = transforms
        self.images_path,self.labels_path = self.readfile(self.data_root,self.mode)
        


    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_path[index]
        label_path  = self.labels_path[index]
        image = np.load(img_path)
        image = image[:,:,:]
        label_ = np.load(label_path)
        label_classes=[]
        for i in range(label_.shape[2]):
            
            label = label_[:,:,i]
            label_FL  = label[label == 2]
            label_TL = label[label == 1]
            if not (label_FL.sum()+label_TL.sum())==0:
                num_percent = label_FL.sum()/(label_FL.sum()+label_TL.sum())
            else:
                num_percent = 0
            if num_percent <= 0.05 :
                label_class = 0
            else:
                label_class = 1
        # image = Image.fromarray(np.uint8(image))
            label_class = int(label_class)
            label_classes.append(label_class)
        sample = {"image": image, "label":label_classes}
        if not self.transforms ==None:
            sample = self.transforms(sample)
            

        # sample = {"image": image, "label":label_class}
        return sample
    
    def __len__(self):
        return len(self.images_path)


    def readfile(self,root,mode):
        if mode == None:
            print('error: please set mode')
        if mode =="train":
            data_path = root + '/train'
        elif mode =='val':
            data_path = root + '/test'
        elif mode == 'test':
            data_path = root + '/test'

        image_files =[]
        label_files =[]

        images_path = data_path + '/image'
        labels_path = data_path + '/label'
        for cur_file_image in sorted(os.listdir(images_path)):
            image_files.append(os.path.join(images_path,cur_file_image))
            label_files.append(os.path.join(labels_path,cur_file_image.split('_')[0]+"_segmentation_"+cur_file_image.split('_')[1]\
        +"_"+ cur_file_image.split('_')[2]))

        return image_files, label_files
    # def labeljudge(label):

# train_data = Arotacls("data/data_classfication/2D",mode = "train")
# train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
# for i,sample in  enumerate(train_dataloader):
#     b = i