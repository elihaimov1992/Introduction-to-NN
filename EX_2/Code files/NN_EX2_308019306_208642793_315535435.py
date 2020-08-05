import numpy as np
import math
from random import random
import matplotlib.pyplot as plt



def loadingNeurons(size,sq,r,shape):
    if (shape == "line"):
        neurons = np.zeros((size, 2))
        s = r / len(neurons)
        first = -r / 2
        for i in range(size):
            neurons[i] = np.array([first, 0])
            first = first + s
        return neurons
    elif shape == "circle":
        neurons = np.zeros((size, 2))
        pi = math.pi
        for i in range(len(neurons)):
            neurons[i][0] = math.cos(2 * pi / len(neurons) * i) * r / 2
            neurons[i][1] = math.sin(2 * pi / len(neurons) * i) * r / 2
        return neurons
        neurons = np.zeros((size, 2))
    elif shape =="5x5topology":  
        neurons = np.zeros((size, 2))  
        s = (r / 2) / (sq - 1)
        Xpoint = -r / 4
        Ypoint = -r / 4
        lines = int(size / sq)
        for i in range(lines):
            for j in range(sq):
                neurons[i * sq + j] = np.array([Xpoint, Ypoint])
                Xpoint = Xpoint + s
            Xpoint = -r / 4
            Ypoint = Ypoint + s
        return neurons


def coord(arr, shape):
    if shape == "5x5topology":
        x = np.array([])
        y = np.array([])
        i = 0
        while i < len(arr):
            x = np.append(x, np.array(arr[i:i + 5, 0]), 0)
            y = np.append(y, np.array(arr[i:i + 5, 1]), 0)
            i = i + 5
        i = 0
        while i < 5:
            j = 0
            colX = np.array([])
            colY = np.array([])
            size = int(len(arr) / 5)
            while j < size:
                colX = np.append(colX, np.array([arr[j * size + i, 0]]), 0)
                colY = np.append(colY, np.array([arr[j * size + i, 1]]), 0)
                j = j + 1
            x = np.append(x, colX, 0)
            y = np.append(y, colY, 0)
            i = i + 1
        return x, y
    elif shape == "circle":
        x = arr[:, 0]
        y = arr[:, 1]
        return x, y
    elif shape == "line":    
         x = arr[:, 0]
         y = arr[:, 1]
         return x, y

def cal_Distance(data, n):
    ans = math.sqrt(math.pow(data[0] - n[0], 2) + math.pow(data[1] - n[1], 2))
    return ans

def winnerNeuronCoord(data, neurons):
    num = math.inf
    neuron_ind =  -1
    for index in range(len(neurons) - 1):
        dis = cal_Distance(data, neurons[index])
        if dis < num:
            num = dis
            neuron_ind = index
    return neuron_ind

def topologicalNeighbor(winner_neuron, n, s):
    distance = cal_Distance(winner_neuron, n)
    ans = math.exp(- (math.pow(distance, 2) / (2 * math.pow(s, 2))))
    return ans

# this function is to:
# update neuron place.
def neuronNewCoord(data, n, a, h):
    new_coord = n + a * h * (data - n)
    return new_coord

def newSigma(s, itr, start):
    ans = s * math.exp(-itr / start)
    return ans

def newAlphe(a, itr, start):
    ans = a * math.exp(-itr / start)
    return ans

def Algorithm(p, n, ep, r):
    a = 0.01
    b = r / 2 + 0.0001
    start = ep / math.log(ep)
    for rounds in range(ep):
        alpha = newAlphe(a, rounds, ep)
        sigma = newSigma(b, rounds, start)
        for point in p:
            winnerNeuron_ind = winnerNeuronCoord(point, n)
            for neurons in range(len(n)):
                dis = cal_Distance(n[winnerNeuron_ind],n[neurons])
                if dis < sigma:
                    h = topologicalNeighbor(n[winnerNeuron_ind],n[neurons],sigma)
                    n[neurons] = neuronNewCoord(point,n[neurons],alpha, h)


r = 2
itr = 1000
x = 0
y = 0
sq = 5
number_of_Neurons = 30
dataShape = "circle"
neuronsShape = "circle"


# scattering data points in circle
if dataShape == "circle":
    circle_r = 4
    circle_amount = 100
    circle__x =0
    circle__y = 0
    size = circle_amount
    answer = np.zeros((size, 2))
    for i in range(size):
        r = circle_r * math.sqrt(random())
        s = random() * 2 * math.pi
        x = circle__x + r * math.cos(s)
        y = circle__y + r * math.sin(s)
        answer[i] = np.array([x,y])
#scattering data points in ring
elif dataShape == "ring":
    size = 100
    answer = np.zeros((size, 2))
    for i in range(size):
        x=0
        y=0
        while (abs(x)<2 and abs(y) < 2):
               r = 4 * math.sqrt(random())
               s = random() * 2 * math.pi
               x= 0 + r * math.cos(s)
               y = 0 + r*math.sin(s)
        answer[i] = np.array([x,y])
if neuronsShape == "line":
    neurons =loadingNeurons(number_of_Neurons,0, 2,"line")
elif neuronsShape == "circle":
    neurons = loadingNeurons(number_of_Neurons,0, 2,"circle")
elif neuronsShape == "5x5topology":
    neurons = loadingNeurons(25, sq, 2,"5x5topology")
Algorithm(answer, neurons, itr, 2)
x_values, y_values = coord(neurons, neuronsShape)
fig, ax = plt.subplots()
plt.scatter(answer[:, 0], answer[:, 1], color='black', marker='.', label='points')
plt.scatter(neurons[:, 0], neurons[:, 1], color='orange', marker='o', label='neurons')
if neuronsShape == "5x5topology":
    x_values = np.array_split(x_values, 10)
    y_values = np.array_split(y_values, 10)
    for val in range(10):
       plt.plot(x_values[val], y_values[val], color='orange', linewidth=1.0)
else:
    plt.plot(x_values, y_values, color='orange', linewidth=1.0)

if dataShape == "circle":
    c1 = plt.Circle((x, y), r, color='white', fill=False)
    ax.add_artist(c1)
elif dataShape == "ring":
    c1 = plt.Circle((x, y), 2, color='white', fill=False)
    c2 = plt.Circle((x, y), 2 * 2, color='white', fill=False)
    ax.add_artist(c1)
    ax.add_artist(c2)
plt.show()