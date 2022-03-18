# train.py
#!/usr/bin/env	python3

""" train Conv-GRU using pytorch
author zhaozhan
"""

import os
import argparse
import time
import logging
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from dataset.transform import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset.dataset_seq import Arotacls_seq
from network.convgru import Res18ConvGRUm1
# from utils.log import logger

num_workers = 2
model_path = './model/convlstm/'
DATE_FORMAT = '%b_%d_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
MILESTONES = [60,140,200]

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, samples in enumerate(train_dataloader):

        if len(args.gpus)>0:
            labels = samples['label'].cuda()
            images = samples['image'].cuda()

        optimizer.zero_grad()
        outputs = net(images)
        outputs = outputs[:,0]
        labels = labels.float()
        y = torch.sigmoid(outputs)
        # y = y.cpu().data.numpy()
        loss = loss_function(y, labels)
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

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    logging.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    val_loss = 0.0 # cost function error
    correct = 0.0
    tp =0.0
    fp =0.0
    fn =0.0
    tn =0.0

    for samples in val_dataloader:

        if len(args.gpus)>0:
            labels = samples['label'].cuda()
            images = samples['image'].cuda()


        outputs = net(images)
        outputs = outputs[:,0]
        labels = labels.float()
        y = torch.sigmoid(outputs)
        # y = y.cpu().data.numpy()
        loss = loss_function(y, labels)

        val_loss += loss.item()
        zero = torch.zeros_like(y)
        one = torch.ones_like(y)
        preds =torch.where(y>0.5, one,zero)
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

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', val_loss / len(val_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(val_dataloader.dataset), epoch)

    return correct.float() / len(val_dataloader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'resnet34_convgru', help='net type')
    parser.add_argument('-gpus', type=list, default=[0,1,2,3], help='use gpu ')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-data_root', type=str, default='./data/2D_seq3_s5/', help='data path root')
    parser.add_argument('-epoch', type=int, default= 200, help='the epoch num')
    parser.add_argument('-checkpoint_path',type= str,default='./output/' )
    parser.add_argument('-log_dir',type= str ,default= './output/')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=os.path.join(
            args.log_dir, args.net, str(TIME_NOW)+str(args.batch_size)))
    snapshot_path = os.path.join(args.log_dir, args.net, str(TIME_NOW)+str(args.batch_size))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    ####   data preprocessing:
    logging.info("Loading dataset...")
    transform =transforms.Compose([
                    CenterCrop((96,96,3)),
                    AddDimToTensor()])                 
    train_data = Arotacls_seq(args.data_root,transforms=transform,mode="train")
    val_data = Arotacls_seq(args.data_root,transforms=transform,mode="val")

    batch_size = args.batch_size if len(args.gpus) == 0 else args.batch_size*len(args.gpus)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logging.info('train dataset len: {}'.format(len(train_dataloader.dataset)))

    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logging.info('val dataset len: {}'.format(len(val_dataloader.dataset)))




    net = Res18ConvGRUm1(1,(12,12),512,[512],[(3,3)],1,torch.cuda.FloatTensor,True,True,False)

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
        checkpoint_path = os.path.join(args.checkpoint_path, args.net, str(TIME_NOW)+str(args.batch_size))

    #use tensorboard
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
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
        if epoch > MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            logging.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % 25:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            logging.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
