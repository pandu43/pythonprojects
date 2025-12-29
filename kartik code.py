### Training

import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm
import multiprocessing
from functools import partial
import os

# Make sure folder exists
os.makedirs('models-SC', exist_ok=True)

BSL = 2048

class Quantize(object):
    def __call__(self, img):
        return torch.div(torch.round(torch.mul(img, BSL // 2)), BSL // 2)

# Resizing the images from 28x28 to 14x14 and Quantizing the pixel values to 16/17 levels
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(14, antialias=True),
    Quantize(),
])

# Load the MNIST dataset
train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test = datasets.MNIST(root='./data', train=False, transform=transform)

# Create data loaders
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 14 * 14)  # Flatten
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class A1(object):
    def __init__(self, width_factor=2, rounding_rate=0.5, n_values=255):
        self.width_factor = width_factor
        self.rounding_rate = rounding_rate
        self.n_values = n_values

    def __call__(self, module):
        for param in module.parameters():
            # calculate the limit for weight values based on the width factor and standard deviation
            # n_lim = param.data.std().item() * self.width_factor
            n_lim = 0.5
            # restrict the range of weight values
            param.data.clamp_(-n_lim, n_lim)

            # quantized weights
            w_q = torch.round(param.data * self.n_values / n_lim) * n_lim / self.n_values
            # put weights closer to its quantized version based on the rounding rate
            param.data = param.data + self.rounding_rate * (w_q - param.data)


model = MLP()

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

# exponential decay of 'width_factor' from 6 to 2 across 'EPOCHS'
width_factors = np.linspace(6, 2, epochs)
# increase of 'rounding_rate' from 0.0 to 1.0 in logistic function across 'EPOCHS'
rounding_rates = np.linspace(0.0, 1.0, epochs)

# Training
accuracies = []
losses = []

for epoch in range(epochs):
    model.train()
    for (data, target) in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()

    # Applying A1
    a1 = A1(width_factor=width_factors[epoch], rounding_rate=rounding_rates[epoch], n_values=BSL // 2)
    model.apply(a1)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    accuracies.append(accuracy)
    losses.append(loss.item())
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), f'models-SC/model_BSL_{BSL}_epoch_{epochs}.pth')
### Inference

def SNG(x, n, probs=None, iterations=1):
    v = int(((x + 1) * n) // 2)  # Number of 1s; 0<= v <=n
    obtained = v

    y = np.zeros((iterations, n), dtype=int)

    for i in range(iterations):
        y[i][np.random.choice(np.arange(n), obtained, replace=False)] = 1

    return y

def Btanh(r, t):
    # r is the number of states in the up/down counter (e.g., 40 states in the example)
    # t is the bipolar stochastic input bit-streams

    Smax = r - 1  # Max state of the up/down counter
    Shalf = r / 2  # Half of the states (center point)
    S = Shalf  # Initial state is the center state

    n = t.shape[0]  # n is the number of bit-streams (10 in this case)
    bsl = t.shape[1]

    output_bitstream = []  # Store the resulting bit-stream after applying Btanh

    for i in range(bsl):
        # Count the number of 1s in the current column of the accumulated result
        V = 2 * sum(t[:, i]) - n  # t[i] is the count of 1s in the ith column

        # Update the state of the up/down counter
        S = S + V
        if S > Smax:
            S = Smax
        elif S < 0:
            S = 0
        # Generate the output bit based on the current state of the counter
        if S > Shalf:
            output_bitstream.append(1)
        else:
            output_bitstream.append(0)
    # The final stochastic bit-stream representing tanh()
    result = np.array(output_bitstream)
    res_val = (2 * np.sum(result) / len(result)) - 1

    return res_val

def find_r(n, s):
    q = 1.835 * ((2 * n) ** (-0.5552))
    r_prime = ((2 * (1 - s) * (n - 1)) / (s * (1 - q))) + 2 * n
    r = 2 * np.round(r_prime / 2).astype(np.int64)
    return r

def ctoDN(x):
    """ Converts the Stochastic Bit Stream to a Digital Number """
    return (2 * np.sum(x) / x.shape[1]) - x.shape[0]

def matrixMultiplication(a, b, probs, n, iterations=1):
    # 'iterations' number of 'a' Stochastic Bit Streams
    x = np.zeros((iterations, a.shape[0], n), dtype=int)

    for i in range(a.shape[0]):
        x[:, i] = SNG(a[i], n, probs, iterations)

    if a.ndim == 1 and b.ndim == 1:
        c = 0
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], n), dtype=int)
        for i in range(b.shape[0]):
            y[:, i] = SNG(b[i], n, probs, iterations)

    elif a.ndim == 1 and b.ndim == 2:
        c = np.zeros(b.shape[0])
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], b.shape[1], n), dtype=int)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                y[:, i, j] = SNG(b[i, j], n, probs, iterations)

    for i in range(iterations):
        if a.ndim == 1 and b.ndim == 1:
            c += Btanh(r,
                       np.array(np.logical_not(np.logical_xor(x[i], y[i])), dtype=int))

        if a.ndim == 1 and b.ndim == 2:
            for j in range(b.shape[0]):
                c[j] += Btanh(r, np.array(np.logical_not(np.logical_xor(x[i], y[i][j])), dtype=int))

    return c / iterations


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def MLPSN(x, modelDN, probs, n):
    # Add 1 to the end of the input vector
    x = np.append(x, 1)
    x = matrixMultiplication(x, modelDN['fc1'] * 2, probs, n=n)

    # x = sigmoid(x)
    # x = tanh(x)

    x = np.append(x, 1)
    x = matrixMultiplication(x, modelDN['fc2'] * 2, probs, n=n)

    return x


class Quantize(object):
    def __init__(self, n=512):
        self.n = n

    def __call__(self, img):
        return torch.div(torch.round(torch.mul(img, self.n)), self.n)


def inference(batch, modelDN, probs, n):
    correct = 0
    total = 0
    for i, (img, label) in tqdm.tqdm(enumerate(batch), total=len(batch)):
        output = MLPSN(np.array(img.view(14 * 14)), modelDN, probs, n=n)
        predicted = np.argmax(output)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        # if i % 10 == 0:
        #     print(f'Accuracy SN | {i}: ', correct / total)
    return correct, total

BSL = 2048

probs = None

model = torch.load(f'models-SC/model_BSL_2048_epoch_100.pth')

# Concatenating the weights and biases of a layer
model['fc1'] = torch.cat(
    (model['fc1.weight'], model['fc1.bias'].reshape(-1, 1)), dim=1)
model['fc2'] = torch.cat(
    (model['fc2.weight'], model['fc2.bias'].reshape(-1, 1)), dim=1)

modelDN = {}
for key, value in model.items():
    modelDN[key] = value.detach().numpy()

# Resizing the images from 28x28 to 14x14 and Quantizing the pixel values to 16/17 levels
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(14, antialias=True),
    Quantize(n=BSL / 2),
])
# Load the MNIST dataset
test = datasets.MNIST(root='./data', train=False, transform=transform)
num_cores = multiprocessing.cpu_count()
# test = torch.utils.data.random_split(test, [100] * 100)[0]
# get 100 samples from each class
# test = torch.utils.data.Subset(test, np.concatenate([np.where(test.targets == i)[0][:100] for i in range(10)]))
chunk_size = len(test) // num_cores
chunks = torch.utils.data.random_split(
    test, [chunk_size] * (num_cores - 1) + [len(test) - chunk_size * (num_cores - 1)])
chunks = [torch.utils.data.DataLoader(
    chunk, shuffle=False) for chunk in chunks]

f = partial(inference, modelDN=modelDN, probs=probs, n=BSL)
with multiprocessing.Pool(num_cores) as pool:
    results = pool.map(f, chunks)

correct, total = zip(*results)
print(f"Final Accuracy SN: {sum(correct) / sum(total)} | n: {BSL}")


