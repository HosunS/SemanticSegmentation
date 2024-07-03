import torch
import pytorch_lightning as pl
from torch.optim import Adam , SGD
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from src.models.supervised.unet import UNet
from src.models.supervised.deeplabv3 import DeepLabV3
from src.models.supervised.dilatedunet import DilatedUNet
from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.resnet_transfer import FCNResnetTransfer

import numpy as np
import torch

class CBSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, class_counts, beta=0.9997):
        super(CBSoftmaxCrossEntropyLoss, self).__init__()
        self.class_counts = class_counts
        self.beta = beta
        self.class_weights = self.calculate_class_weights(class_counts, beta)
        
    def calculate_class_weights(self, class_counts, beta):
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(class_counts)  # normalize weights to sum to number of classes
        return torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, logits, targets):
        device = targets.device  # get the device of the targets tensor
        class_weights = self.class_weights.to(device)  # move class_weights to the same device
        
        logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        targets = targets.view(-1)
        
        weights = class_weights[targets]
        loss = F.cross_entropy(logits, targets, reduction='none')
        loss = loss * weights
        return loss.mean()
    
    
    ### scheduler similar to documentation    
class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power for base_lr in self.base_lrs]


class ESDSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}, freeze_backbone=True,max_iters=10000, power=0.9):
        '''
        Constructor for ESDSegmentation class.
        '''
        # call the constructor of the parent class
        super(ESDSegmentation,self).__init__()
        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()
        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        self.max_iters = max_iters  # maximum number of iterations
        self.power = power  # power for poly scheduler
        # self.l1_lambda = 1e-5
        # # if the model type is unet, initialize a unet as self.model
        if model_type == "UNet":
            self.model = UNet(in_channels,out_channels,**model_params)
        elif model_type == "DilatedUNet":
            self.model = DilatedUNet(in_channels,out_channels,**model_params)
        elif model_type == "DeepLabV3":
            self.model = DeepLabV3(in_channels,out_channels,**model_params)
        elif model_type =="SegmentationCNN":
            self.model = SegmentationCNN(self.in_channels,self.out_channels,**model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels,out_channels,**model_params)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # # Freeze layers if required
        # if freeze_backbone and model_type in ["DeepLabV3"]:
        #     for param in self.model.deeplabv3.backbone.parameters():
        #         param.requires_grad = False
        
        # # Unfreeze classifier layers for training
        # for param in self.model.deeplabv3.classifier.parameters():
        #     param.requires_grad = True
        
        
        # initialize the accuracy metrics for the semantic segmentation task
        # self.accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=out_channels)
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=out_channels)
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=out_channels)
        # self.train_f1 = torchmetrics.classification.BinaryF1Score()
        # self.val_f1 = torchmetrics.classification.BinaryF1Score()
        # self.jaccard_index = torchmetrics.JaccardIndex(task="multiclass",num_classes=out_channels)
        # self.auc=torchmetrics.classification.MulticlassAUROC(num_classes=out_channels)
        # self.intersection_union = torchmetrics.detection.IntersectionOverUnion()
        
        #test using weights taken from classweights.py
        # class_weights = torch.tensor([0.02435222, 0.01886662 ,0.22759958 ,0.72918158], dtype=torch.float32) 
        # self.loss = nn.CrossEntropyLoss(weight=class_weights)
        
        #class balanced loss
        # class_counts = [6318, 8155,  676,  211]
        # beta = .9997
        # self.loss = CBSoftmaxCrossEntropyLoss(class_counts, beta)
        
        #softmax gets applied in crossentropyloss?
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, X):
        # evaluate self.model
        return self.model(X.to(torch.float32))
    
    def calculate_l1_norm(self):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return l1_norm
    
    ##might need to log it in to wandb here, we can see later.
    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        # pretty sure that batch returns a tuple with sat_img / mask
        sat_img, mask = batch
        # evaluate batch
        outputs = self.forward(torch.nan_to_num(sat_img.type(torch.float32)))
        pred = torch.argmax(outputs, dim=1)
        mask = mask.to(torch.int64)
  
        #f1 
        train_f1 = self.train_f1(pred, mask)
        self.log("train_f1", train_f1, prog_bar=True, on_step=False, on_epoch=True)
    
        loss = self.loss(outputs, mask)
        # l1_norm = self.calculate_l1_norm()
        # loss += self.l1_lambda * l1_norm  # Add L1 regularization term
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # return loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img , mask = batch
        # evaluate batch for validation
        outputs = self.forward(torch.nan_to_num(sat_img.type(torch.float32)))
        # get the class with the highest probability
        # print(outputs)

        pred = torch.argmax(outputs,dim=1)
        mask = mask.to(torch.int64)
    
        #f1
        val_f1 = self.val_f1(pred, mask)
        self.log("val_f1", val_f1, prog_bar=True,on_step=False)
        
        # return validation loss 
        loss = self.loss(outputs,mask)
        self.log("val_loss", loss, prog_bar=True,on_step=False)
        
        # return validation loss 
        return loss
    
    def configure_optimizers(self):
        # initialize optimizer
        #test with L2 regularization
        optimizer = Adam(self.parameters(),lr = self.hparams.learning_rate)
        scheduler = PolyLRScheduler(optimizer, max_iter=self.hparams.max_iters, power=self.hparams.power)
        # optimizer = Adam(self.parameters(),lr = self.hparams.learning_rate,weight_decay=1e-3)
                
        # return optimizer
        return [optimizer],[scheduler]
        

