import torch
import torch.nn as nn
from transfer_losses import TransferLoss
from torchvision import models


class mybackbone(nn.Module):
    def __init__(self):
        super(mybackbone, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        return x2, x
    
    def output_num(self):
        return self._feature_dim

    

class myTransferNet(nn.Module):
    def __init__(self, num_class, transfer_loss1='mmd', transfer_loss2 = 'adv', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(myTransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = mybackbone()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss1 = transfer_loss1
        self.transfer_loss2 = transfer_loss2
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.layer2tf = nn.Sequential(*[nn.Conv2d(in_channels=128, out_channels=bottleneck_width, kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=28)])
            
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss1_args = {
            "loss_type": self.transfer_loss1,
            "max_iter": max_iter,
            "num_class": num_class}
        transfer_loss2_args = {
            "loss_type": self.transfer_loss2,
            "max_iter": max_iter,
            "num_class": num_class}
        self.adapt_loss1 = TransferLoss(**transfer_loss1_args)
        self.adapt_loss2 = TransferLoss(**transfer_loss2_args)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, source, target, source_label):
        source2,source = self.base_network(source)
        target2,target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        source2 = self.layer2tf(source2)
        target2 = self.layer2tf(target2)
        source2 = torch.squeeze(torch.squeeze(source2, -1), -1)
        target2 = torch.squeeze(torch.squeeze(target2, -1), -1)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer
        kwargs = {}
        transfer_loss1 = self.adapt_loss1(source2, target2, **kwargs)
        transfer_loss2 = self.adapt_loss2(source, target, **kwargs)
        
        return clf_loss, transfer_loss1, transfer_loss2
        
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        params.append(
                {'params': self.adapt_loss2.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params
    
    def predict(self, x):
        features2, features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf
    
    def epoch_based_processing(self, *args, **kwargs):
        pass