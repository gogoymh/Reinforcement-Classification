import torch
import torch.nn as nn

class block_a(nn.Module):
    def __init__(self):
        super(block_a, self).__init__()        
        self.conv = nn.Conv2d(3,3,3,1,1,bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out) + x        
        return out

class block_b(nn.Module):
    def __init__(self):
        super(block_b, self).__init__()        
        self.conv = nn.Conv2d(3,3,3,1,1,bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out) + x    
        return out

class classifier_a(nn.Module):
    def __init__(self):
        super(classifier_a, self).__init__()        
        self.fc1 = nn.Linear(3*32*32, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = x.view(-1,3*32*32)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class classifier_b(nn.Module):
    def __init__(self):
        super(classifier_b, self).__init__()        
        self.fc1 = nn.Linear(3*32*32, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = x.view(-1,3*32*32)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Environment:
    def __init__(self, model, path, dataset, device):
        ## ---- build model ---- ##
        self.device = device
        self.model = model().to(self.device)
        if self.device == "cuda:0":
            self.checkpoint = torch.load(path)
        else:
            self.checkpoint = torch.load(path, map_location=lambda storage, location: 'cpu')
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model_module_list = list(self.model.modules())
        
        ## ---- get linear filter index ---- ##
        self.model_linear_layer = []
        #self.model_bn_layer = []
        for i, m in enumerate(self.model_module_list):
            if type(m) == nn.Linear:
                self.model_linear_layer.append(i)
            #elif type(m) == nn.BatchNorm2d:
            #    self.model_bn_layer.append(i)
        
        ## ---- save datset ---- ##
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        
        ## ---- build block network and classifier ---- ##
        self.block = block_a().to(self.device)
        self.block_module_list = list(self.block.modules())
        self.classifier = classifier_a().to(self.device)
        self.classifier_module_list = list(self.classifier.modules())
        
        ## ---- get conv filter and linear filter index of each network ---- ##
        self.block_conv_layer = []
        for i, m in enumerate(self.block_module_list):
            if type(m) == nn.Conv2d:
                self.block_conv_layer.append(i)
        
        self.classifier_linear_layer = []
        for i, m in enumerate(self.classifier_module_list):
            if type(m) == nn.Linear:
                self.classifier_linear_layer.append(i)
        
        ## ---- copy parameter from model to classifier ---- ##
        for i in range(2):
            self.classifier_module_list[self.classifier_linear_layer[i]].weight.data = self.model_module_list[self.model_linear_layer[i]].weight.data.clone()
            self.classifier_module_list[self.classifier_linear_layer[i]].bias.data = self.model_module_list[self.model_linear_layer[i]].bias.data.clone()
            
            
    def reset(self):        
        state, y = self.dataset.__iter__().next()
        self.current_state = state
        self.current_state_idx = 0
        self.ground_truth = y
        
        return [self.current_state, self.current_state_idx], self.ground_truth
    
    def step(self, action):
        self.block_module_list[self.block_conv_layer[0]].weight.data = action.clone()
        #self.block_module_list[self.block_bn_layer[0]].weight.data = self.model_module_list[self.model_bn_layer[self.current_state_idx]].weight.data.clone()
        #self.block_module_list[self.block_bn_layer[0]].bias.data = self.model_module_list[self.model_bn_layer[self.current_state_idx]].bias.data.clone()
        
        state = self.block(self.current_state)
        self.current_state = state
        self.current_state_idx += 1
        
        return [self.current_state, self.current_state_idx]

    def get_reward(self, final_state):
        output = self.classifier(final_state)
        loss = self.criterion(output, self.ground_truth.long().to(self.device))
        
        return -loss.item()


























