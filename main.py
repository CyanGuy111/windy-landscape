import numpy as np
import os
import csv

lSize = [784, 128, 10]
rate = 0.1
batchsize = 100
epoch = 100
testsize = 500
np.random.seed(1)

def load_data(filename='data.csv'):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]
    data = np.array(data)
    inp = data[:, 1:]
    y = data[:, 0].astype(int)
    true = np.zeros((len(y), 10))
    true[np.arange(len(y)), y] = 1
    return inp, true

def save_params(w1, b1, w2, b2, filename='params.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(w1)
        writer.writerow(b1.flatten())
        writer.writerows(w2)
        writer.writerow(b2.flatten())
    return

def load_params(filename='params.csv'):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [[float(x) for x in row] for row in reader]
    i = 0
    w1 = np.array(data[i:i+lSize[0]])
    i += lSize[0]
    b1 = np.array(data[i]).reshape(1, lSize[1])
    i += 1
    w2 = np.array(data[i:i+lSize[1]])
    i += lSize[1]
    b2 = np.array(data[i]).reshape(1, lSize[2])
    i += 1
    return w1, b1, w2, b2

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def loss(pred, true):
    n = pred.shape[0]
    return -np.sum(true * np.log(pred + 1e-9)) / n

def one_hot(y):
    Y = np.zeros(10)
    Y[y] = 1
    return Y

def train(inp, true, load_ex = True):
    if load_ex and os.path.exists('params.csv'):
        w1, b1, w2, b2 = load_params()
    else:
        w1 = np.random.randn(lSize[0], lSize[1]) * 2
        b1 = np.zeros((1, lSize[1]))
        w2 = np.random.randn(lSize[1], lSize[2]) * 2
        b2 = np.zeros((1, lSize[2]))
    
    for ex in range(epoch):
        for i in range(0, inp.shape[0] - 1000, batchsize):
            inp_b = inp[i:i + batchsize]
            true_b = true[i:i + batchsize]

            z1 = inp_b @ w1 + b1
            a1 = ReLU(z1)
            z2 = a1 @ w2 + b2
            a2 = softmax(z2)

            l = loss(a2, true_b)

            dz2 = (a2 - true_b) / batchsize
            dw2 = a1.T @ dz2
            db2 = np.sum(dz2, axis = 0, keepdims = True)

            da1 = dz2 @ w2.T
            dz1 = da1 * dReLU(z1)
            dw1 = inp_b.T @ dz1
            db1 = np.sum(dz1, axis = 0, keepdims = True)

            w1 -= rate * dw1
            w2 -= rate * dw2
            b1 -= rate * db1
            b2 -= rate * db2

            # print(f"Progress {(i + 1) / (inp.shape[0] - 1000) * 100 :.1f}")

        print(f"Epoch {ex + 1}, Loss: {l:.6f}")
    save_params(w1, b1, w2, b2)

def test(inp, true):
    w1, b1, w2, b2 = load_params()
    score = 0
    for i in range(9500, 9500 + testsize):
        inp_b = inp[i]
        true_b = true[i]

        z1 = inp_b @ w1 + b1
        a1 = ReLU(z1)
        z2 = a1 @ w2 + b2
        a2 = softmax(z2)
        if(np.argmax(true_b) == np.argmax(a2)):
            score += 1
    print(f"{(score / testsize * 100):.2f}%")

def run(inp):
    w1, b1, w2, b2 = load_params()
    z1 = inp @ w1 + b1
    a1 = ReLU(z1)
    z2 = a1 @ w2 + b2
    a2 = softmax(z2)
    pred = np.argmax(a2)
    return pred
inp, true = load_data()

train(inp, true)
test(inp, true)