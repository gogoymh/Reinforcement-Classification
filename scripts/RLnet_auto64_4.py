import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import random
import numpy as np
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
        
        self.layer1 = self.make_conv_layer(3)
        self.layer1_bn = self.make_bn_layer(3)
        self.layer2 = self.make_conv_layer(3)
        self.layer2_bn = self.make_bn_layer(3)
        self.layer3 = self.make_conv_layer(3)
        self.layer3_bn = self.make_bn_layer(3)
        self.layer4 = self.make_conv_layer(3)
        self.layer4_bn = self.make_bn_layer(3)
        
        self.layer5 = self.make_conv_layer(3)
        self.layer5_bn = self.make_bn_layer(3)
        self.layer6 = self.make_conv_layer(3)
        self.layer6_bn = self.make_bn_layer(3)
        self.layer7 = self.make_conv_layer(3)
        self.layer7_bn = self.make_bn_layer(3)
        self.layer8 = self.make_conv_layer(3)
        self.layer8_bn = self.make_bn_layer(3)
        
        self.layer9 = self.make_conv_layer(3)
        self.layer9_bn = self.make_bn_layer(3)
        self.layer10 = self.make_conv_layer(3)
        self.layer10_bn = self.make_bn_layer(3)
        self.layer11 = self.make_conv_layer(3)
        self.layer11_bn = self.make_bn_layer(3)
        self.layer12 = self.make_conv_layer(3)
        self.layer12_bn = self.make_bn_layer(3)
        
        self.layer13 = self.make_conv_layer(3)
        self.layer13_bn = self.make_bn_layer(3)
        self.layer14 = self.make_conv_layer(3)
        self.layer14_bn = self.make_bn_layer(3)
        self.layer15 = self.make_conv_layer(3)
        self.layer15_bn = self.make_bn_layer(3)
        self.layer16 = self.make_conv_layer(3)
        self.layer16_bn = self.make_bn_layer(3)
        
        self.conv_dropout = nn.Dropout2d(0.1)
        self.conv1 = nn.Conv2d(3,3,4,2,1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3,3,4,2,1)
        self.bn2 = nn.BatchNorm2d(3)
        
        self.fc_dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(3*8*8, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        
        self.relu = nn.LeakyReLU()
        #self.sigmoid = nn.Sigmoid()
        
    def make_conv_layer(self, channel):
        
        layer = [nn.Conv2d(channel,channel,3,1,1), nn.Dropout2d(0.1)]
        
        return nn.Sequential(*layer)
    
    def make_bn_layer(self, channel):
        
        layer = [nn.BatchNorm2d(channel), nn.Softmax2d()]
        
        return nn.Sequential(*layer)
        
    def forward(self, x, train=True):
        cum_feat = x/2
        
        out1 = self.layer1(x) + cum_feat
        out1 = self.layer1_bn(out1)
        cum_feat = (cum_feat + out1)/2
        
        out2 = self.layer2(out1) + cum_feat
        out2 = self.layer2_bn(out2)
        cum_feat = (cum_feat + out2)/2
        
        out3 = self.layer3(out2) + cum_feat
        out3 = self.layer3_bn(out3)
        cum_feat = (cum_feat + out3)/2
        
        out4 = self.layer4(out3) + cum_feat
        out4 = self.layer4_bn(out4)
        cum_feat = (cum_feat + out4)/2
        
        out5 = self.layer5(out4) + cum_feat
        out5 = self.layer5_bn(out5)
        cum_feat = (cum_feat + out5)/2
        
        out6 = self.layer6(out5) + cum_feat
        out6 = self.layer6_bn(out6)
        cum_feat = (cum_feat + out6)/2
        
        out7 = self.layer7(out6) + cum_feat
        out7 = self.layer7_bn(out7)
        cum_feat = (cum_feat + out7)/2
        
        out8 = self.layer8(out7) + cum_feat
        out8 = self.layer8_bn(out8)
        cum_feat = (cum_feat + out8)/2
        
        out9 = self.layer9(out8) + cum_feat
        out9 = self.layer9_bn(out9)
        cum_feat = (cum_feat + out9)/2
        
        out10 = self.layer10(out9) + cum_feat
        out10 = self.layer10_bn(out10)
        cum_feat = (cum_feat + out10)/2
        
        out11 = self.layer11(out10) + cum_feat
        out11 = self.layer11_bn(out11)
        cum_feat = (cum_feat + out11)/2
        
        out12 = self.layer12(out11) + cum_feat
        out12 = self.layer12_bn(out12)
        cum_feat = (cum_feat + out12)/2
        
        out13 = self.layer13(out12) + cum_feat
        out13 = self.layer13_bn(out13)
        cum_feat = (cum_feat + out13)/2
        
        out14 = self.layer14(out13) + cum_feat
        out14 = self.layer14_bn(out14)
        cum_feat = (cum_feat + out14)/2
        
        out15 = self.layer15(out14) + cum_feat
        out15 = self.layer15_bn(out15)
        cum_feat = (cum_feat + out15)/2
        
        out16 = self.layer16(out15) + cum_feat
        out16 = self.layer16_bn(out16)        
        
        out = self.conv1(out16) ## check!!
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
        
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc_dropout(out)
        
        if train:
            return F.log_softmax(out, dim=0)
        else:
            idx = 1
            image1 = x[idx].clone().detach().cpu()
            image1[0] = (image1[0] * 0.2023) + 0.4914
            image1[1] = (image1[1] * 0.1994) + 0.4822
            image1[2] = (image1[2] * 0.2010) + 0.4465
            image1 = image1.numpy().transpose(1,2,0)
            image2 = out1[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image3 = out2[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image4 = out3[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image5 = out4[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            
            image6 = out5[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image7 = out6[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image8 = out7[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image9 = out8[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            
            image10 = out9[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image11 = out10[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image12 = out11[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image13 = out12[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            
            image14 = out13[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image15 = out14[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image16 = out15[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            image17 = out16[idx].clone().detach().cpu().numpy().transpose(1,2,0)
            
            plt.imshow(image1)
            plt.show()
            plt.close()
            
            plt.imshow(np.concatenate((image2,image3,image4,image5),axis=1))
            plt.show()
            plt.close()
            
            plt.imshow(np.concatenate((image6,image7,image8,image9),axis=1))
            plt.show()
            plt.close()
            
            plt.imshow(np.concatenate((image10,image11,image12,image13),axis=1))
            plt.show()
            plt.close()
            
            plt.imshow(np.concatenate((image14,image15,image16,image17),axis=1))
            plt.show()
            plt.close()
            
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
print("="*100)
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
        print("="*100)
        accuracy = 0
        with torch.no_grad():
            model.eval()
            correct = 0
            for batch_idx, (x, y) in enumerate(test_loader):
                if batch_idx == 0:
                    output = model(x.float().to(device),train=False)
                
                else:
                    output = model(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
            accuracy = correct / len(test_loader.dataset)
        print("="*100)
        print("Accuracy is %f" % accuracy)
        print("="*100)
        model.train()

import platform
save_path = "/DATA/ymh/rlcl/results/RLnet_auto16_4.pkl" if platform.system() == "Linux" else "C:/유민형/개인 연구/Reinforcement Classification/results/RLnet_auto16_4.pkl"

torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[epoch].item()}, save_path)

#   "C:/유민형/개인 연구/Reinforcement Classification/results/RLnet_auto64.pkl"














