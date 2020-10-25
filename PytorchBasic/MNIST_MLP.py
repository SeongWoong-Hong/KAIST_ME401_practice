#%%
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#%%
root = './data'
if not os.path.exists(root):
    os.mkdir(root)
#%%
train_set = dset.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
test_set = dset.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
#%%
batch_size = 100
total_epoch = 10
learning_rate = 0.01
use_cuda = torch.cuda.is_available()
#%%
train_loader = torch.utils.data.DataLoader(  # [Question 2] What is 'torch.utils.data.DataLoader' used for?
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

#%%
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)  # [Question 3] What is 'nn.Linear' used for?
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # [Question 4] (4-1)What is 'view' used for? (4-2)What does '-1' mean?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"
#%%
model = MLPNet()
if use_cuda:
    model = model.cuda()
#%%
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
# %%
for epoch in range(total_epoch):
    # trainning
    total_loss = 0
    total_batch = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()

        out = model(x)
        loss = criterion(out,target)
        total_loss += loss.item()
        total_batch += 1
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'
                  .format(epoch, batch_idx + 1, total_loss / total_batch))
    # testing
    total_loss = 0
    total_batch = 0
    correct_cnt = 0
    total_cnt = 0

    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()

        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum().item()

        total_loss += loss.item()
        total_batch += 1

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'
                  .format(epoch, batch_idx + 1, total_loss / total_batch, correct_cnt * 1.0 / total_cnt))