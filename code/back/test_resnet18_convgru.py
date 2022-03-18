#test.py
#!/usr/bin/env python3

""" test neuron network performace
print on test dataset
of a model
author zhan
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset.transform import *
from dataset.dataset_seq import Arotacls_seq
from network.convgru import *

num_workers = 2
model_path = './output/AD_classfication/resnet101/train01'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'resnet18-convgru', help='net type')
    parser.add_argument('-gpus', type=list, default=[0], help='use gpu ')
    parser.add_argument('-batch_size', type=int, default=512, help='batch size for dataloader')
    parser.add_argument('-data_root', type=str, default='./data/2D_seq3_s5', help='data path root')
    parser.add_argument('-weights', type=str, default='./model/resnet18_convgru.pth', help='the weights file you want to test')
    parser.add_argument('-result',type= str ,default= './result/')
    args = parser.parse_args()
    result_path = args.result +args.net + '.txt'

    transform =transforms.Compose([
                    CenterCrop((96,96,3)),
                    AddDimToTensor()])                  
    test_data = Arotacls_seq(args.data_root,transforms=transform,mode="test")
    test_dataloader = DataLoader(test_data,args.batch_size, shuffle=False, num_workers=num_workers)
    net = ResNetConvGRU(1,(12,12),512,[512],[(3,3)],1,torch.cuda.FloatTensor,True,True,False)
    net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()
    net.load_state_dict(torch.load(args.weights))
    # print(net)
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    confusionmatrix =np.zeros((2,2))

    with torch.no_grad():
        for n_iter, samples  in enumerate(test_dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_dataloader)))

            if len(args.gpus) >0 :

                labels = samples['label'].cuda()
                images = samples['image'].cuda()


            outputs = net(images)
            outputs = outputs[:,0]
            labels = labels.float()
            y = torch.sigmoid(outputs)

            zero = torch.zeros_like(y)
            one = torch.ones_like(y)
            preds =torch.where(y>0.5,one,zero)
            y_pred= preds.cpu().numpy()
            y_true = labels.cpu().numpy()

            confusionmatrix_slice=confusion_matrix(y_true, y_pred,labels=[0,1])
            confusionmatrix = confusionmatrix+confusionmatrix_slice


        fp = confusionmatrix[0][1]
        fn = confusionmatrix[1][0]
        tp = confusionmatrix[1][1]
        tn = confusionmatrix[0][0]
        precision = tp / (tp+fp)  # 查准率
        recall = tp / (tp+fn)  # 查全率
        accuracy = (tp+tn)/(tp+tn+fp+fn) #准确率
        f1score = 2*precision*recall/(precision+recall) #F1 score
    with open(result_path,"w") as f:
        f.write('混淆矩阵:\n')
        f.write('TP:%d\t'%(tp))
        f.write('FP:%d\n'%(fp))
        f.write('FN:%d\t'%(fn))
        f.write('TN:%d\n'%(tn))
        f.write('Precision:%4f\t'%(precision))
        f.write('Recall:%4f\n'%(recall))
        f.write('Accuracy:%4f\n'%(accuracy))
        f.write('F1 score:%4f\n'%(f1score))
        

        