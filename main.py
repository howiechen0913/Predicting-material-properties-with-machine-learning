# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 23:30:22 2021

@author: user
"""
from __future__ import print_function, division
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import math
import numpy as np
from torchvision import datasets, transforms, models
import torch 
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import csv
import time
import itertools
import json
import torch.nn.functional as F
import sys


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

use_gpu = torch.cuda.is_available()
torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser(description='MLP train or test battery material')

parser.add_argument('--train-data', default='./data/train_data2.csv', help='train data path')

parser.add_argument('--val-data', default='./data/val_data2.csv', help='val data path')

parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')

parser.add_argument('--StepLR', default=100, nargs='+', type=int,
                    metavar='N', help='Step for scheduler (default: '
                                      '[100])')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')

parser.add_argument('--cuda', default='cuda:0', type=str,
                    help='cuda number, (default: cuda:0)')

args = parser.parse_args(sys.argv[1:])

#args.cuda = not args.disable_cuda and torch.cuda.is_available()

classes_types=['trigonal','triclinic','tetragonal','monoclinic','orthorhombic','hexagonal','cubic']
#classes_types = ['unstable', 'unstable_RT', 'stable']


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 計算出的混淆矩陣值
    - classes : 類別名稱
    - normalize : True:顯示百分比, False:顯示個數
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('/home/howchen' +  '/' + "_confusion_matrix_8_7.jpg", dpi=400)
    plt.savefig('./result/img/confusion_matrix.jpg',dpi=400)

bandgap_list = []

class train_ElementDataset(Dataset):
    def __init__(self):
        data_file = args.train_data
        element_vector = []
        Class = []
        with open (data_file, 'r',newline=('')) as csv_date:
            csvReader = csv.DictReader(csv_date)
            for x in csvReader:
                returns = x['element_vector']
                returns = returns.split()
                vector = list(map(float, returns))
                element_vector.append(vector)
                
                if args.task == 'classification':
                    crystal_class = x['class']
                    crystal_class = int(crystal_class)
                    Class.append(crystal_class)
                    
                else:
                    bandgap = x['bandgap']
                    bandgap = float(bandgap)
                    bandgap_list.append(bandgap)
                    Class.append(bandgap)

            element_vector_data = torch.Tensor(element_vector)
            Class_data = torch.Tensor(Class)
        self.data = element_vector_data
        self.label = Class_data

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

class val_ElementDataset(Dataset):
    def __init__(self):
        data_file = args.val_data
        element_vector = []
        Class = []
        with open (data_file, 'r',newline=('')) as csv_date:
            csvReader = csv.DictReader(csv_date)
            for x in csvReader:
                returns = x['element_vector']
                returns = returns.split()
                vector = list(map(float, returns))
                element_vector.append(vector)
                
                if args.task == 'classification':
                    crystal_class = x['class']
                    crystal_class = int(crystal_class)
                    Class.append(crystal_class)
                    
                else:
                    bandgap = x['bandgap']
                    bandgap = float(bandgap)
                    Class.append(bandgap)

            element_vector_data = torch.Tensor(element_vector)
            Class_data = torch.Tensor(Class)
        self.data = element_vector_data
        self.label = Class_data

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)


class FCNet(nn.Module):

    def __init__(self, num_classes=1):
        if args.task == 'classification':
            num_classes = 7
        else:
            num_classes = 1
        super(FCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(143, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    global args
    device = torch.device(args.cuda)
    train_dataloader = DataLoader(dataset=train_ElementDataset(),batch_size=args.batch_size,shuffle='True')
    train_dataset_number = len(train_ElementDataset().data)
    val_dataloader = DataLoader(dataset=val_ElementDataset(),batch_size=args.batch_size,shuffle='True')
    val_dataset_number  = len(val_ElementDataset().data)
    model = FCNet().cuda(device)
    
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0, 'std': 1})

    else:
        normalizer = Normalizer(torch.Tensor(bandgap_list))  
        
    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss(reduction='mean')
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    epochs = args.epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.StepLR,gamma=0.1)
    train_model(model, train_dataloader, val_dataloader, train_dataset_number, val_dataset_number, normalizer, criterion, device, optimizer,exp_lr_scheduler, num_epochs=epochs)
    
def train_model(model, train_dataloader, val_dataloader, train_dataset_number, val_dataset_number, normalizer, criterion, devices, optimizer, scheduler, num_epochs=25):
	since = time.time()
	best_model_wts = model.state_dict()
	best_acc = 0.0
	best_loss = 1e5

	n = 0
	train_loss = []
	train_acc = []
    
	val_acc = []
	val_loss = []
	
	bast_val_predict = 0
	val_save_value = 0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		
		for phase in ['train', 'test']:
			if phase == 'train':
				scheduler.step()
				model.train(True)
			else:
				model.train(False)
            
			running_loss = 0.0
			running_corrects = 0
			n = 0
			if phase == 'train':
				for i, data in enumerate(train_dataloader):
					inputs, labels = data
					bs, vectorsize = inputs.size()

					inputs = inputs.view(-1, vectorsize)

					inputs, labels = Variable(inputs.cuda(devices)), Variable(labels.cuda(devices))
					labels = torch.as_tensor(labels,dtype = torch.long)

					optimizer.zero_grad()

					outputs = model(inputs)
                    
					original_label = labels

					if args.task == 'classification':
						norm_labels = labels
					else:
						norm_labels = normalizer.norm(labels)
                        
					denorm_loss = criterion(normalizer.denorm(outputs), original_label)


					regularization_loss = 0
					for param in model.parameters():
						regularization_loss += torch.sum(abs(param))
					save_loss = denorm_loss 
					if args.task == 'classification':
						_, preds = torch.max(outputs, 1)
						running_corrects += (preds == original_label.data).sum().item()
						
					loss = criterion(outputs, norm_labels)
					loss = loss + 0.00001 * regularization_loss
					loss.backward()
					optimizer.step()
					
					scheduler.step()
                    
					running_loss += save_loss.data
					n+=1

				epoch_loss = running_loss / n
				epoch_acc = running_corrects / train_dataset_number

				train_loss.append(float(epoch_loss))
				train_acc.append(epoch_acc)
                
				if args.task == 'classification':
					print('{} Loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
				else:
					print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                
			if phase == 'test':
                
				val_predict = []
				val_trage = []

				for i, data in enumerate(val_dataloader):

					inputs, labels = data
					bs, vectorsize = inputs.size()
					inputs = inputs.view(-1, vectorsize)

					inputs, labels = Variable(inputs.cuda(devices)), Variable(labels.cuda(devices))
					labels = torch.as_tensor(labels,dtype = torch.long)
					outputs = model(inputs)
					
					original_label = labels
                    
					if args.task == 'classification':
						norm_labels = labels
					else:
						norm_labels = normalizer.norm(labels)
                        
					loss = criterion(normalizer.denorm(outputs), original_label)
					
					if args.task == 'classification':
						_, preds = torch.max(outputs, 1)
						running_corrects += (preds == original_label.data).sum().item()
						val_predict.append(preds)
					else:
						val_predict.append(normalizer.denorm(outputs))
                        
					val_trage.append(original_label.tolist())

					running_loss += loss.data
					n+=1

				epoch_acc = running_corrects / val_dataset_number
				epoch_loss = running_loss / n
				val_loss.append(float(epoch_loss))
				val_acc.append(epoch_acc)
				
				if args.task == 'classification':
					print('{} Loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
				else:
					print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                
				if phase == 'test' and epoch_loss < best_loss:
					best_loss = epoch_loss
					best_model_wts = model.state_dict()
					bast_val_predict = np.array(val_predict,dtype=object)
					bast_val_predict = bast_val_predict.flatten()
					val_save_value = np.array(val_trage,dtype=object)
					val_save_value = val_save_value.flatten()
                                
		time_elapsed = time.time() - since
		print('Training complete in {:0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		if args.task == 'classification':
			print('Best val Acc: {:4f}'.format(best_acc))
            
		print('Best val loss: {:4f}'.format(best_loss))
		model.load_state_dict(best_model_wts)
	

	with open('./result/json/train_loss.json','w') as train_loss_json:
		train_loss_2 = json.dumps(train_loss)
		json.dump(train_loss_2, train_loss_json)
        
	with open('./result/json/val_loss.json','w') as val_loss_json:
		val_loss_2 = json.dumps(val_loss)
		json.dump(val_loss_2, val_loss_json)

	with open('./result/csv/val_predict_value.csv','w',newline=('')) as val_predict_value_write:
		csv_predict_value_writer = csv.writer(val_predict_value_write)

		for x,y in zip(val_save_value,bast_val_predict):
			for i,n in zip(x,y):
				csv_predict_value_writer.writerow([float(i),float(n)])

	if args.task == 'classification':
		nb_classes = 7
		confusion_matrix = np.zeros((nb_classes, nb_classes))
		for x,y in zip(val_save_value,bast_val_predict):
			for i,n in zip(x,y):
				confusion_matrix[i, n] += 1
        
		plot_confusion_matrix(confusion_matrix, classes=classes_types, normalize=True, title='Normalized confusion matrix')
        
	model.load_state_dict(best_model_wts)
	torch.save(model, './model/best_model.pkl')
	torch.save(model.state_dict(), './model/best_model_p.pkl')


if __name__ == '__main__':
    main()

