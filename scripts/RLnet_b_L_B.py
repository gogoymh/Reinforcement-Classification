import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import random
import numpy as np
import timeit

randomseed = 0
torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
print("="*100)
print("Random seed is %d" % randomseed)
############################################################################################################
class ReinforceNet(nn.Module):
    def __init__(self):
        super(ReinforceNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn3 = nn.BatchNorm2d(3)
        self.conv4 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn4 = nn.BatchNorm2d(3)
        
        self.conv5 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn5 = nn.BatchNorm2d(3)
        self.conv6 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn6 = nn.BatchNorm2d(3)
        self.conv7 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn7 = nn.BatchNorm2d(3)
        self.conv8 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)
        
        self.conv9 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn9 = nn.BatchNorm2d(3)
        self.conv10 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn10 = nn.BatchNorm2d(3)
        self.conv11 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn11 = nn.BatchNorm2d(3)
        self.conv12 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn12 = nn.BatchNorm2d(3)
        
        self.conv13 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn13 = nn.BatchNorm2d(3)
        self.conv14 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn14 = nn.BatchNorm2d(3)
        self.conv15 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn15 = nn.BatchNorm2d(3)
        self.conv16 = nn.Conv2d(3,3,3,1,1, bias=False)
        self.bn16 = nn.BatchNorm2d(3)
        
        self.fc1 = nn.Linear(3*32*32, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        out1 = self.relu(x)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1) + x
        
        out2 = self.relu(out1)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2) + out1
        
        out3 = self.relu(out2)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3) + out2
        
        out4 = self.relu(out3)
        out4 = self.conv4(out4)
        out4 = self.bn4(out4) + out3
        
        out5 = self.relu(out4)
        out5 = self.conv5(out5)
        out5 = self.bn5(out5) + out4
        
        out6 = self.relu(out5)
        out6 = self.conv6(out6)
        out6 = self.bn6(out6) + out5
        
        out7 = self.relu(out6)
        out7 = self.conv7(out7)
        out7 = self.bn7(out7) + out6
        
        out8 = self.relu(out7)
        out8 = self.conv8(out8)
        out8 = self.bn8(out8) + out7
        
        out9 = self.relu(out8)
        out9 = self.conv9(out9)
        out9 = self.bn9(out9) + out8
        
        out10 = self.relu(out9)
        out10 = self.conv10(out10)
        out10 = self.bn10(out10) + out9
        
        out11 = self.relu(out10)
        out11 = self.conv11(out11)
        out11 = self.bn11(out11) + out10
        
        out12 = self.relu(out11)
        out12 = self.conv12(out12)
        out12 = self.bn12(out12) + out11
        
        out13 = self.relu(out12)
        out13 = self.conv13(out13)
        out13 = self.bn13(out13) + out12
        
        out14 = self.relu(out13)
        out14 = self.conv14(out14)
        out14 = self.bn14(out14) + out13
        
        out15 = self.relu(out14)
        out15 = self.conv15(out15)
        out15 = self.bn15(out15) + out14
        
        out16 = self.relu(out15)
        out16 = self.conv16(out16)
        out16 = self.bn16(out16) + out15
        
        out = out16.view(-1,3*32*32)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
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
#print(model)
#print("How many total parameters       | %d" % cnt)
print("="*100)
#################################################################################################################

losses = torch.zeros((1000))

for epoch in range(1000):
    start = timeit.default_timer()
    for x, y in train_loader:
        
        optimizer.zero_grad()
        
        output = model(x.float().to(device))
        
        loss = criterion(output, y.long().to(device))
        
        loss.backward()
        
        optimizer.step()
        
        losses[epoch] += loss.item()
    
    losses[epoch] /= len(train_loader)
    stop = timeit.default_timer()
    print("[Epoch:%d] Loss is %f" % ((epoch+1), losses[epoch].item()))
    print("Time for single epoch is",stop-start, "seconds")
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

torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[epoch].item()}, "C:/유민형/개인 연구/Reinforcement Classification/results/RLnet_b_L_B.pkl")

#"/home/super/ymh/rlcl/results/RLnet_b_L_B.pkl"















