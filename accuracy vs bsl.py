import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import multiprocessing
from functools import partial
import pandas as pd

############################################# Inference Functions #######################################

def SNG(x, n, probs=None, iterations=1):
    v = int(((x + 1) * n) // 2)  # Number of 1s
    y = np.zeros((iterations, n), dtype=int)
    for i in range(iterations):
        y[i][np.random.choice(np.arange(n), v, replace=False)] = 1
    return y

def Btanh(r, t):
    Smax = r - 1
    Shalf = r / 2
    S = Shalf
    n, bsl = t.shape
    output_bitstream = []

    for i in range(bsl):
        V = 2 * sum(t[:, i]) - n
        S = min(max(S + V, 0), Smax)
        output_bitstream.append(1 if S > Shalf else 0)

    result = np.array(output_bitstream)
    return (2 * np.sum(result) / len(result)) - 1

def find_r(n, s):
    q = 1.835 * ((2 * n) ** (-0.5552))
    r_prime = ((2 * (1 - s) * (n - 1)) / (s * (1 - q))) + 2 * n
    r = 2 * np.round(r_prime / 2).astype(np.int64)
    return r

def matrixMultiplication(a, b, probs, n, iterations=1):
    x = np.zeros((iterations, a.shape[0], n), dtype=int)
    for i in range(a.shape[0]):
        x[:, i] = SNG(a[i], n, probs, iterations)

    if b.ndim == 1:
        c = 0
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], n), dtype=int)
        for i in range(b.shape[0]):
            y[:, i] = SNG(b[i], n, probs, iterations)

        for i in range(iterations):
            c += Btanh(r, np.array(np.logical_not(np.logical_xor(x[i], y[i])), dtype=int))
    else:
        c = np.zeros(b.shape[0])
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], b.shape[1], n), dtype=int)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                y[:, i, j] = SNG(b[i, j], n, probs, iterations)

        for i in range(iterations):
            for j in range(b.shape[0]):
                c[j] += Btanh(r, np.array(np.logical_not(np.logical_xor(x[i], y[i][j])), dtype=int))

    return c / iterations

def MLPSN(x, modelDN, probs, n):
    x = np.append(x, 1)  # Add bias
    x = matrixMultiplication(x, modelDN['fc1'] * 2, probs, n=n)
    x = np.append(x, 1)  # Add bias for second layer
    x = matrixMultiplication(x, modelDN['fc2'] * 2, probs, n=n)
    return x

def inference(batch, modelDN, probs, n):
    correct = 0
    total = 0
    for img, label in batch:
        output = MLPSN(np.array(img.view(14*14)), modelDN, probs, n=n)
        predicted = np.argmax(output)
        total += 1
        correct += int(predicted == label.item())
    return correct, total

############################################# Load Model #######################################
BSL_default = 2048
probs = None
# Load PyTorch model
model = torch.load(f'models-SC/model_BSL_2048_epoch_100.pth')

# Concatenate weights and biases
model['fc1'] = torch.cat((model['fc1.weight'], model['fc1.bias'].reshape(-1, 1)), dim=1)
model['fc2'] = torch.cat((model['fc2.weight'], model['fc2.bias'].reshape(-1, 1)), dim=1)

# Convert to numpy
modelDN = {k: v.detach().numpy() for k, v in model.items()}

############################################# Load MNIST Dataset #######################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(14, antialias=True),
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

num_cores = multiprocessing.cpu_count()
chunk_size = len(test_dataset) // num_cores
chunks = torch.utils.data.random_split(
    test_dataset, [chunk_size]*(num_cores-1) + [len(test_dataset) - chunk_size*(num_cores-1)]
)
chunks = [DataLoader(chunk, shuffle=False) for chunk in chunks]

############################################# Run BSL Sweep #######################################

BSL_list = [2**i for i in range(3, 11)]  # 8 → 2048
acc_list = []

for BSL in BSL_list:
    f = partial(inference, modelDN=modelDN, probs=probs, n=BSL)
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(f, chunks)

    correct, total = zip(*results)
    acc = sum(correct) / sum(total)
    acc_list.append(acc)
    print(f"BSL={BSL}, Accuracy={acc:.4f}")

# Create a DataFrame
df = pd.DataFrame({
    'BSL': BSL_list,
    'Accuracy': acc_list
})
# Save to CSV
df.to_csv('accuracy_vs_BSL.csv', index=False)
print("CSV file saved as 'accuracy_vs_BSL.csv'")

############################################# Plot Results #######################################

plt.figure(figsize=(7,5))
plt.plot(BSL_list, acc_list, marker='o')
plt.xscale("log", base=2)
plt.xlabel("Bitstream Length (BSL)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs BSL for SC MLP")
plt.grid(True)
plt.tight_layout()
plt.show()