import json

from matplotlib import pyplot as plt

with open('info.txt', 'r') as file:
    y_best = []
    y_worst = []
    y_mean = []
    iteration = 0
    inp = json.load(file)
    for line in inp:
        it1, it2, it3 = line
        y_best.append(it1)
        y_worst.append(it2)
        y_mean.append(it3)
    x = range(len(y_best))
    plt.plot(x, y_best, label='best fit')
    plt.plot(x, y_worst, label='worst fit')
    plt.plot(x, y_mean, label='mean fit')

    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    # giving a title to my graph
    plt.title('learning curve')
    
    plt.show()
