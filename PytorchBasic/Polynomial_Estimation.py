#%%
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
#%%
POLY_DEGREE = 4
torch.manual_seed(2020)
W_target = torch.randn(POLY_DEGREE + 1, 1) * 5
#%%
def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result
#%%
print('==> The real function you should approximate:\t' + poly_desc(W_target[:-1].view(-1), W_target[-1]))
#%%
def make_features(x):
    """Builds features i.e. a matrix with columns [x^4, x^3, x^2, x^1]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** (POLY_DEGREE+1-i) for i in range(1, POLY_DEGREE+1)], 1)
#%%
def f(x, W):
    """Approximated function."""
    return x.mm(W[:-1]) + W[-1]
#%%
def get_dataset(dataset_size):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(dataset_size)
    x = make_features(random)
    y = f(x, W_target)
    dataset = list(zip(x, y))
    return dataset
#%%
dataset = get_dataset(200) # you can make as many as dataset as you want
num_epochs = 1000
batch_size = 20
learning_rate = 0.05
criterion = nn.SmoothL1Loss()
#%%
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(POLY_DEGREE, 1)

        # For fixing the initial weights and bias
        self.fc.weight.data.fill_(0.)
        self.fc.bias.data.fill_(0.)
    def forward(self, x):
        output = self.fc(x)
        return output
#%%
def fit(model,loader,criterion,learning_rate,num_epochs):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_values=[]

    for epoch in range(num_epochs):

        running_loss=0.0
        for i, data in enumerate(loader):
            if torch.cuda.is_available():
                x = data[0].type(torch.FloatTensor).cuda()
                y = data[1].type(torch.FloatTensor).cuda()
            else:
                x = data[0].type(torch.FloatTensor)
                y = data[1].type(torch.FloatTensor)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_values.append(running_loss)

    return loss_values
#%%
def fit2(W, loader,criterion,learning_rate,num_epochs):
    loss_values = []
    for iter in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(loader):
            x, y_tar = data
            y = f(x, W)
            loss = criterion(y, y_tar)
            loss.backward()
            with torch.no_grad():
                W -= learning_rate * W.grad
                W.grad.zero_()
            running_loss += loss.item()

        loss_values.append(running_loss)

    return loss_values

#%%
net = Net().cuda() if torch.cuda.is_available() else Net()
W_est = torch.randn(POLY_DEGREE + 1, 1, requires_grad=True)
#%%
loss_values1 = fit(net, dataset_loader,criterion,learning_rate,num_epochs)
loss_values2 = fit2(W_est, dataset_loader,criterion,learning_rate,num_epochs)
#%%
print('==> Actual function:\t' + poly_desc(W_target[:-1].view(-1), W_target[-1]))
print('==> Learned function:\t' + poly_desc(net.fc.weight.data.view(-1), net.fc.bias.data))
print('==> Estimated function:\t' + poly_desc(W_est[:-1].view(-1), W_est[-1]))