#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

from dataset.dataset import Arotacls
from network.resnet import *

num_workers = 2
model_path = './output/AD_classfication/resnet101/train01'
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
MILESTONES = [60, 120, 160]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'resnet50', help='net type')
    parser.add_argument('-gpus', type=list, default=[0], help='use gpu ')
    parser.add_argument('-batch_size', type=int, default=512, help='batch size for dataloader')
    # parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    # parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    # parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-data_root', type=str, default='./data/2D', help='data path root')
    # parser.add_argument('-epoch', type=int, default= 200, help='the epoch num')
    parser.add_argument('-weights', type=str, default='./model/resnet50.pth', help='the weights file you want to test')
    parser.add_argument('-result',type= str ,default= './result/')
    args = parser.parse_args()
    result_path = args.result +args.net + '.txt'

    transform =transforms.Compose([
                    # transforms.Scale(120),
                    transforms.CenterCrop(96),
                    transforms.ToTensor()])                 
    test_data = Arotacls(args.data_root,transforms=transform,mode="test")
    test_dataloader = DataLoader(test_data,args.batch_size, shuffle=False, num_workers=num_workers)
    net = resnet50(pretrained=False, modelpath=model_path, num_classes=1)
    loss_function = nn.BCELoss()
    net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()
    net.load_state_dict(torch.load(args.weights))
    # print(net)
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    confusionmatrix =np.zeros((2,2))

    with torch.no_grad():
        for n_iter, (images, labels)  in enumerate(test_dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_dataloader)))

            if len(args.gpus) >0 :

                labels = labels.cuda()
                images = images.cuda()


            outputs = net(images)
            outputs = outputs[:,0]
            labels = labels.float()
            y = torch.sigmoid(outputs)
            loss = loss_function(y, labels)

            test_loss += loss.item()
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
        