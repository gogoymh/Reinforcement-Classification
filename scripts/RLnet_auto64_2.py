import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn.functional as F

#import random
#import numpy as np
#import timeit
'''
randomseed = 0
torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
print("="*100)
print("Random seed is %d" % randomseed)
'''
############################################################################################################
class ReinforceNet(nn.Module):
    def __init__(self):
        super(ReinforceNet, self).__init__()
        
        self.layer1 = self.make_layer()
        self.layer2 = self.make_layer()
        self.layer3 = self.make_layer()
        self.layer4 = self.make_layer()
        
        self.layer5 = self.make_layer()
        self.layer6 = self.make_layer()
        self.layer7 = self.make_layer()
        self.layer8 = self.make_layer()
        
        self.layer9 = self.make_layer()
        self.layer10 = self.make_layer()
        self.layer11 = self.make_layer()
        self.layer12 = self.make_layer()
        
        self.layer13 = self.make_layer()
        self.layer14 = self.make_layer()
        self.layer15 = self.make_layer()
        self.layer16 = self.make_layer()
        '''
        self.layer17 = self.make_layer()
        self.layer18 = self.make_layer()
        self.layer19 = self.make_layer()
        self.layer20 = self.make_layer()
        
        self.layer21 = self.make_layer()
        self.layer22 = self.make_layer()
        self.layer23 = self.make_layer()
        self.layer24 = self.make_layer()
        
        self.layer25 = self.make_layer()
        self.layer26 = self.make_layer()
        self.layer27 = self.make_layer()
        self.layer28 = self.make_layer()
        
        self.layer29 = self.make_layer()
        self.layer30 = self.make_layer()
        self.layer31 = self.make_layer()
        self.layer32 = self.make_layer()
        
        self.layer33 = self.make_layer()
        self.layer34 = self.make_layer()
        self.layer35 = self.make_layer()
        self.layer36 = self.make_layer()
        
        self.layer37 = self.make_layer()
        self.layer38 = self.make_layer()
        self.layer39 = self.make_layer()
        self.layer40 = self.make_layer()
        
        self.layer41 = self.make_layer()
        self.layer42 = self.make_layer()
        self.layer43 = self.make_layer()
        self.layer44 = self.make_layer()
        
        self.layer45 = self.make_layer()
        self.layer46 = self.make_layer()
        self.layer47 = self.make_layer()
        self.layer48 = self.make_layer()
        
        self.layer49 = self.make_layer()
        self.layer50 = self.make_layer()
        self.layer51 = self.make_layer()
        self.layer52 = self.make_layer()
        
        self.layer53 = self.make_layer()
        self.layer54 = self.make_layer()
        self.layer55 = self.make_layer()
        self.layer56 = self.make_layer()
        
        self.layer57 = self.make_layer()
        self.layer58 = self.make_layer()
        self.layer59 = self.make_layer()
        self.layer60 = self.make_layer()
        
        self.layer61 = self.make_layer()
        self.layer62 = self.make_layer()
        self.layer63 = self.make_layer()
        self.layer64 = self.make_layer()
        '''
        
        self.conv_dropout = nn.Dropout2d(0.1)
        self.conv1 = nn.Conv2d(3,3,4,2,1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3,3,4,2,1)
        self.bn2 = nn.BatchNorm2d(3)
        
        self.fc_dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(3*8*8, 100)
        self.fc2 = nn.Linear(100, 10)
        
        self.relu = nn.LeakyReLU()
        
    def make_layer(self):
        
        layer = [nn.LeakyReLU(), nn.Conv2d(3,3,3,1,1,bias=False), nn.Dropout2d(0.1), nn.BatchNorm2d(3)]
        
        return nn.Sequential(*layer)
    
    def forward(self, x):
        
        def cumulative_output(layer, sum_feature, input_feature):
            
            output = layer(input_feature) + sum_feature
            cumulative_feature = (sum_feature + output)/2
            
            return output, cumulative_feature
        
        cum_feat = x/2
        
        out1, cum_feat = cumulative_output(self.layer1, cum_feat, x)
        out2, cum_feat = cumulative_output(self.layer2, cum_feat, out1)
        out3, cum_feat = cumulative_output(self.layer3, cum_feat, out2)
        out4, cum_feat = cumulative_output(self.layer4, cum_feat, out3)
        
        out5, cum_feat = cumulative_output(self.layer5, cum_feat, out4)
        out6, cum_feat = cumulative_output(self.layer6, cum_feat, out5)
        out7, cum_feat = cumulative_output(self.layer7, cum_feat, out6)
        out8, cum_feat = cumulative_output(self.layer8, cum_feat, out7)
        
        out9, cum_feat = cumulative_output(self.layer9, cum_feat, out8)
        out10, cum_feat = cumulative_output(self.layer10, cum_feat, out9)
        out11, cum_feat = cumulative_output(self.layer11, cum_feat, out10)
        out12, cum_feat = cumulative_output(self.layer12, cum_feat, out11)
        
        out13, cum_feat = cumulative_output(self.layer13, cum_feat, out12)
        out14, cum_feat = cumulative_output(self.layer14, cum_feat, out13)
        out15, cum_feat = cumulative_output(self.layer15, cum_feat, out14)
        out16, cum_feat = cumulative_output(self.layer16, cum_feat, out15)
        '''
        out17, cum_feat = cumulative_output(self.layer17, cum_feat, out16)
        out18, cum_feat = cumulative_output(self.layer18, cum_feat, out17)
        out19, cum_feat = cumulative_output(self.layer19, cum_feat, out18)
        out20, cum_feat = cumulative_output(self.layer20, cum_feat, out19)
        
        out21, cum_feat = cumulative_output(self.layer21, cum_feat, out20)
        out22, cum_feat = cumulative_output(self.layer22, cum_feat, out21)
        out23, cum_feat = cumulative_output(self.layer23, cum_feat, out22)
        out24, cum_feat = cumulative_output(self.layer24, cum_feat, out23)
        
        out25, cum_feat = cumulative_output(self.layer25, cum_feat, out24)
        out26, cum_feat = cumulative_output(self.layer26, cum_feat, out25)
        out27, cum_feat = cumulative_output(self.layer27, cum_feat, out26)
        out28, cum_feat = cumulative_output(self.layer28, cum_feat, out27)
        
        out29, cum_feat = cumulative_output(self.layer29, cum_feat, out28)
        out30, cum_feat = cumulative_output(self.layer30, cum_feat, out29)
        out31, cum_feat = cumulative_output(self.layer31, cum_feat, out30)
        out32, cum_feat = cumulative_output(self.layer32, cum_feat, out31)
        
        out33, cum_feat = cumulative_output(self.layer33, cum_feat, out32)
        out34, cum_feat = cumulative_output(self.layer34, cum_feat, out33)
        out35, cum_feat = cumulative_output(self.layer35, cum_feat, out34)
        out36, cum_feat = cumulative_output(self.layer36, cum_feat, out35)
        
        out37, cum_feat = cumulative_output(self.layer37, cum_feat, out36)
        out38, cum_feat = cumulative_output(self.layer38, cum_feat, out37)
        out39, cum_feat = cumulative_output(self.layer39, cum_feat, out38)
        out40, cum_feat = cumulative_output(self.layer40, cum_feat, out39)
        
        out41, cum_feat = cumulative_output(self.layer41, cum_feat, out40)
        out42, cum_feat = cumulative_output(self.layer42, cum_feat, out41)
        out43, cum_feat = cumulative_output(self.layer43, cum_feat, out42)
        out44, cum_feat = cumulative_output(self.layer44, cum_feat, out43)
        
        out45, cum_feat = cumulative_output(self.layer45, cum_feat, out44)
        out46, cum_feat = cumulative_output(self.layer46, cum_feat, out45)
        out47, cum_feat = cumulative_output(self.layer47, cum_feat, out46)
        out48, cum_feat = cumulative_output(self.layer48, cum_feat, out47)
        
        out49, cum_feat = cumulative_output(self.layer49, cum_feat, out48)
        out50, cum_feat = cumulative_output(self.layer50, cum_feat, out49)
        out51, cum_feat = cumulative_output(self.layer51, cum_feat, out50)
        out52, cum_feat = cumulative_output(self.layer52, cum_feat, out51)
        
        out53, cum_feat = cumulative_output(self.layer53, cum_feat, out52)
        out54, cum_feat = cumulative_output(self.layer54, cum_feat, out53)
        out55, cum_feat = cumulative_output(self.layer55, cum_feat, out54)
        out56, cum_feat = cumulative_output(self.layer56, cum_feat, out55)
        
        out57, cum_feat = cumulative_output(self.layer57, cum_feat, out56)
        out58, cum_feat = cumulative_output(self.layer58, cum_feat, out57)
        out59, cum_feat = cumulative_output(self.layer59, cum_feat, out58)
        out60, cum_feat = cumulative_output(self.layer60, cum_feat, out59)
        
        out61, cum_feat = cumulative_output(self.layer61, cum_feat, out60)
        out62, cum_feat = cumulative_output(self.layer62, cum_feat, out61)
        out63, cum_feat = cumulative_output(self.layer63, cum_feat, out62)
        out64, cum_feat = cumulative_output(self.layer64, cum_feat, out63)
        '''
        
        out = self.relu(out16)
        out = self.conv1(out)
        out = self.conv_dropout(out)
        out = self.bn1(out)
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv_dropout(out)
        out = self.bn2(out)
        
        out = out.view(-1,3*8*8)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.fc_dropout(out)
        
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc_dropout(out)
        
        return F.log_softmax(out, dim=0)
    
##############################################################################################################
train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=64, shuffle=True, pin_memory=True)
                                
test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False, pin_memory=True)
#################################################################################################################
criterion = nn.CrossEntropyLoss()

model = ReinforceNet()
device = torch.device("cuda:0")
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

#################################################################################################################
params = list(model.parameters())
cnt = 0
for i in range(len(params)):
    cnt += params[i].data.reshape(-1,).shape[0]
print(model)
print("How many total parameters       | %d" % cnt)
print("="*100)
#################################################################################################################

losses = torch.zeros((3000))

for epoch in range(3000):
    #start = timeit.default_timer()
    for x, y in train_loader:
        
        optimizer.zero_grad()
        
        output = model(x.float().to(device))
        
        loss = criterion(output, y.long().to(device))
        
        loss.backward()
        
        optimizer.step()
        
        losses[epoch] += loss.item()
    
    losses[epoch] /= len(train_loader)
    #stop = timeit.default_timer()
    print("[Epoch:%d] Loss is %f" % ((epoch+1), losses[epoch].item()))
    #print("Time for single epoch is",stop-start, "seconds")
    if (epoch+1) % 10 == 0:     
        accuracy = 0
        with torch.no_grad():
            model.eval()
            correct = 0
            for x, y in test_loader:
                output = model(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
            accuracy = correct / len(test_loader.dataset)
            
        print("Accuracy is %f" % accuracy)
        print("="*100)
        model.train()

import platform
save_path = "/DATA/ymh/rlcl/results/RLnet_auto16_2.pkl" if platform.system() == "Linux" else "C:/유민형/개인 연구/Reinforcement Classification/results/RLnet_auto16_2.pkl"

torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[epoch].item()}, save_path)

#   "C:/유민형/개인 연구/Reinforcement Classification/results/RLnet_auto64.pkl"














