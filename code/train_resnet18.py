# train.py
#!/usr/bin/env	python3

""" train resnet using pytorch
author zhaozhan
"""

import imp
import os
import argparse
import time
import logging
import sys
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset.dataset import Arotacls
from network.resnet import resnet18
import warnings
warnings.filterwarnings("ignore")
# from utils.log import logger

num_workers = 2
DATE_FORMAT = '%b_%d_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
MILESTONES = [60, 120, 170]

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_dataloader):

        if len(args.gpus)>0:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        outputs = outputs[:,0]
        labels = labels.float()
        outputs = torch.sigmoid(outputs)
        # y = y.cpu().data.numpy()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        logging.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(train_dataloader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    logging.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    val_loss = 0.0 # cost function error
    correct = 0.0

    for batch_index,(images, labels) in enumerate(val_dataloader):

        if len(args.gpus)>0:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        outputs = outputs[:,0]
        outputs = torch.sigmoid(outputs)
        labels = labels.float()
        loss = loss_function(outputs, labels)

        val_loss += loss.item()
        zero = torch.zeros_like(outputs)
        one = torch.ones_like(outputs)
        preds =torch.where(outputs>0.5, one,zero)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if len(args.gpus)>0:
    #     logging.info('GPU INFO.....')
    #     logging.info(torch.cuda.memory_summary(), end='')
    logging.info('Evaluating Network.....')
    logging.info('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        val_loss / len(val_dataloader.dataset),
        correct.float() / len(val_dataloader.dataset),
        finish - start
    ))
    # logging.info()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', val_loss / len(val_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(val_dataloader.dataset), epoch)

    return correct.float() / len(val_dataloader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'resnet18-nobg', help='net type')
    parser.add_argument('-gpus', type=list, default=[0,1,2,3,4,5,6,7], help='use gpu ')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-data_root', type=str, default='./data/2D_nobg', help='data path root')
    parser.add_argument('-epoch', type=int, default= 200, help='the epoch num')
    parser.add_argument('-checkpoint_path',type= str,default='./output/' )
    parser.add_argument('-log_dir',type= str ,default= './output/')

    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join(
            args.log_dir, args.net, TIME_NOW))
    snapshot_path = os.path.join(args.log_dir, args.net, TIME_NOW)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    ####   data preprocessing:
    logging.info("Loading dataset...")
    transform =transforms.Compose([
                    # transforms.Scale(120),
                    transforms.CenterCrop(96),
                    transforms.ToTensor()])                 
    train_data = Arotacls(args.data_root,transforms=transform,mode="train")
    val_data = Arotacls(args.data_root,transforms=transform,mode="val")

    batch_size = args.batch_size if len(args.gpus) == 0 else args.batch_size*len(args.gpus)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logging.info('train dataset len: {}'.format(len(train_dataloader.dataset)))

    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logging.info('val dataset len: {}'.format(len(val_dataloader.dataset)))
    
    model_path = os.path.join(args.log_dir, args.net, TIME_NOW)
    net = resnet18(pretrained=False, modelpath=model_path, num_classes=1)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCELoss() 
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(args.checkpoint_path, args.net), fmt=DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(args.checkpoint_path, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(args.checkpoint_path, args.net, TIME_NOW)

    #use tensorboard
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
    #         args.log_dir, args.net, TIME_NOW))
    # input_tensor = torch.Tensor(1, 3, 32, 32)
    if len(args.gpus) > 0:
        gpus = ','.join([str(x) for x in args.gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        args.gpus = tuple(range(len(args.gpus)))
        # logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))
        net = nn.DataParallel(net, device_ids=args.gpus)
        net = net.cuda()
        # input_tensor = input_tensor.cuda()

    # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(args.checkpoint_path, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(args.checkpoint_path, args.net, recent_folder, best_weights)
            logging.info('found best acc weights file:{}'.format(weights_path))
            logging.info('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            logging.info('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(args.checkpoint_path, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(args.checkpoint_path, args.net, recent_folder, recent_weights_file)
        logging.info('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(args.checkpoint_path, args.net, recent_folder))


    for epoch in range(1, args.epoch + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > MILESTONES[0] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            logging.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % 50:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            logging.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
