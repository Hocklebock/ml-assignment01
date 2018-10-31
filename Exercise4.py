import matplotlib.pyplot as plt
from numpy import array, unique, mean, amax, amin, transpose, concatenate, ones, sum, dot, matrix
from numpy.random import randint, uniform

# open file and split header from data
file = open('Fisher.txt', 'r')
lines = file.read().splitlines()
head = lines.pop(0)

# reading data from file into numpy matrix and sort
data = array([[int(num) for num in line.split('\t')] for line in lines])
data = data[data[:, 0].argsort()]
data_by_type = [data[data[:, 0] == type] for type in unique(data[:, 0])]

# compute statistical values
means = [mean(element, axis=0) for element in data_by_type]
maxs = [amax(element, axis=0) for element in data_by_type]
mins = [amin(element, axis=0) for element in data_by_type]

# visualize
f1 = plt.figure(1)
plt.xlabel('pental length')
plt.ylabel('sepal length')
plt.title('Fisher`s Iris Data')
plt.grid(True)
plt.scatter(data_by_type[0][:, 2], data_by_type[0][:, 4], 20, c="g")
plt.scatter(data_by_type[1][:, 2], data_by_type[1][:, 4], 20, c="r")
plt.scatter(data_by_type[2][:, 2], data_by_type[2][:, 4], 20, c="b")
plt.savefig("exercise4c.png")
# plt.show()

# linear regression
# LMS Solving
x = concatenate((data_by_type[0][:, 4], data_by_type[1][:, 4]), axis=0).flatten()
c = concatenate((data_by_type[0][:, 0], data_by_type[1][:, 0]), axis=0).flatten()


def lms(x, c, iterations=1000, eta=1e-5, plotoptions=None):
    w = uniform(low=-1.0, high=1.0, size=2)
    count = 0
    wtrack = []

    while count < iterations:
        index = randint(len(x))
        xi = array([1.0, x[index]])
        ci = c[index]
        yi = dot(w, xi)
        error = ci - yi
        w += eta * error * xi
        wtrack.append(w.flatten())
        count += 1

    if plotoptions:
        conv = sum([(ci - dot(w, [1, xi])) ** 2 for xi, ci in zip(x, c)])
        f2 = plt.figure(2)
        plt.xlabel('iterations')
        plt.ylabel('weight')
        plt.title('convergence')
        wtrack = matrix(wtrack)
        plt.plot(range(count), wtrack[:, 0], c="g")
        plt.plot(range(count), wtrack[:, 1], c="b")
        plt.savefig("convergence4d.png")
        print(conv, w[0], w[1])

    return w


w = lms(x, c, 10000000, 0.0005)

f3 = plt.figure(3)
plt.xlabel('sepal length')
plt.ylabel('class')
plt.title('Fisher`s Iris Data')
plt.grid(True)
plt.scatter(data_by_type[0][:, 4], data_by_type[0][:, 0], 20, c="g")
plt.scatter(data_by_type[1][:, 4], data_by_type[1][:, 0], 20, c="r")
plt.plot(x, w[0] + w[1] * x, c="b")
plt.savefig("exercise4d.png")
plt.show()
