import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import ssl
import math
from scipy.stats import norm
import random

ssl._create_default_https_context = ssl._create_unverified_context


epoch_size = 10000

s_max = 1000.
s_min = 0.

K_max = 1000.
K_min = 0.

r_max = 30.
r_min = 0.

t_max = 50.
t_min = 0.

vol_max = 70.
vol_min = 0.

scale_factor = K_max

def calc_euro_vanilla_call(S, K, r, t, vol ):
    r'''
    S = Spot price of underlying
    K = Strike price
    r = Risk free interest rate
    t = Time to maturity (in years)
    vol = Volatility of underlying 
    '''
    d1 = (math.log((S/K)) + ( (r + ((vol * vol)/2.)) * t )) / (vol * math.sqrt(t))
    d2 = d1 - (vol * math.sqrt(t))
    price = (norm.cdf(d1) * S) - (norm.cdf(d2) * K * (math.exp(-r * t)))
    
    return  price


class EuroVanillaCallDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.random = random.Random()

    def __len__(self):
        return epoch_size

    def __getitem__(self, idx):
        S = random.randint(s_min, s_max - 1) + self.random.random() 
        K = random.randint(K_min, K_max - 1) + self.random.random()
        r = random.randint(r_min, r_max - 1) + self.random.random()
        t = random.randint(t_min, t_max - 1) + self.random.random()
        vol = random.randint(vol_min, vol_max - 1) + self.random.random()

        price = calc_euro_vanilla_call(S, K, r, t, vol)

        S /= s_max
        K /= K_max
        r /= r_max
        t /= t_max
        vol /= vol_max

        price /= scale_factor 
        price = float(price)
        params = torch.tensor([S, K, r, t, vol])

        return   params, price


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.inputs = 5
        self.hidden_count = 512

        self.fc1 = torch.nn.Linear(self.inputs, self.hidden_count)
        self.elu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_count, self.hidden_count)
        self.elu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_count, self.hidden_count)
        self.elu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_count, self.hidden_count)
        self.elu4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(self.hidden_count, 1)
        self.elu5 = torch.nn.ReLU()

    def forward(self, x):
        o = self.fc1(x)
        o = self.elu1(o)
        o = self.fc2(o)
        o = self.elu2(o)
        o = self.fc3(o)
        o = self.elu3(o)
        o = self.fc4(o)
        o = self.elu4(o)
        o = self.fc5(o)

        return o


def get_device():
    use_cuda = torch.cuda.is_available()
    print('use_cuda: {}'.format(use_cuda))
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    
def load_checkpoint(filename):
    model = Model()
    checkpoint= torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model

def validate(val_loader, model, loss_function, device):
    running_loss = 0.0

    model.eval()

    for i, x in enumerate(val_loader):

        params, price = x
        input = params.to(device)
        target = price.to(torch.float32).to(device) * scale_factor

        output = model(input)
        loss = loss_function(output, target)

        running_loss += loss.item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    print (f'Epoch Validation Loss {epoch_loss}')
        

def train(train_loader, model, loss_function, optimizer, epoch, device):

    model.train()
    for i, x in enumerate(train_loader):
        params, price = x
        input = params.to(device)
        target = price.to(torch.float32).to(device) * scale_factor

        output = model(input)
        loss = loss_function(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % epoch_size == 0:
            print(f'Training Loss: {loss}')
            print(f'Analytic Price: {price[0] * scale_factor}. Model price: {output[0]}')

def infer(S, K, r, t, vol, model, device):
    '''   S = Spot price of underlying
    K = Strike price
    r = Risk free interest rate
    t = Time to maturity (in years)
    vol = Volatility of underlying '''

    model.eval()

    S /= s_max
    K /= K_max
    r /= r_max
    t /= t_max
    vol /= vol_max

    params = torch.tensor([S, K, r, t, vol])
    input = params.to(torch.float32).to(device)

    price = model(input)

    return price

def infer_test():
    S = 100
    K = 500
    r = 1
    t = 1
    vol = 0.1

    device = get_device()
    model = load_checkpoint('24.checkpoint.pth.tar')
    model = model.to(device)

    analytic_price = calc_euro_vanilla_call(S, K, r, t, vol)
    model_price = infer(S, K, r, t, vol, model, device)

    print(f'Analytic Price: {analytic_price}. Model price: {model_price}')   


def main():
    device = get_device()
    model = Model()
    model = model.to(device)
    train_dataset = EuroVanillaCallDataset()
    val_dataset = EuroVanillaCallDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle=True,num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle=False,num_workers=1, pin_memory = False )

    loss_function = nn.MSELoss().to(device)

    learning_rate = 0.001
    epochs = 1000000

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(0, epochs):
        print(f'Epoch: {epoch}')
        train(train_loader, model, loss_function, optimizer, epoch, device)
        validate(val_loader, model, loss_function, device)
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()}, filename=f'{epoch}.checkpoint.pth.tar')

if __name__ == '__main__':
    main()
    #infer_test()
