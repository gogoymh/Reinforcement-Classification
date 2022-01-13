############################################################################################################
class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        
        self.conv = nn.Conv2d(3,3,3,1,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out) + x        
        return out

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        
        self.fc1 = nn.Linear(3*32*32, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

############################################################################################################
Environment = DataLoader(
        datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=1, shuffle=True, pin_memory=True)
agent = Agent()

net = block().to(device)
module_list = list(net.modules())

classify = classifier().to(device)
classify.eval()

data_holder = processor()

for j in range(episodes):
    ## ---- get initial state ---- ##
    state, y = Environment.__iter__().next()
    data_holder.state_reciever(state)
    
    for i in range(steps=16):
        ## ---- get action ---- ##
        action = agent.searching_action(j, state) # 3*3*3*3 tensor
        data_holder.action_reciever(action)
        
        ## ---- execute action ---- ##
        module_list[i].weight.data = action.clone().to(device)
        state = net(state)
        data_holder.move_idx()
        data_holder.state_reciever(state)

    ## ---- get reward ---- ##
    output = classify(state)
    pred = output.argmax(1, keepdim=True)
    reward = pred.eq(y.long().to(device).view_as(pred)).sum().item()
    data_holder.reward_reciever(reward)
    
    ## ---- update ---- ##
    a, b, c, d = data_holder.conver_data()
    agent.update_memory_buffer(a,b,c,d)
    agent.update_main_network()
    if len(agent.memory.buffer) > agent.warmup:
        if (j+1) % 10 == 0:
            agent.update_target_network()
            print("[Episode: %d] [Reward: %f] [delta: %f]" % ((j+1), reward, agent.delta))