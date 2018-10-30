import matplotlib.pyplot as plt
from random import randint
from numpy import array, unique, mean, amax, amin, concatenate, ones, sum, dot

#open file and split header from data
file = open('Fisher.txt', 'r')
lines = file.read().splitlines()
head = lines.pop(0)

#reading data from file into numpy matrix and sort
data = array([[int(num) for num in line.split('\t')] for line in lines])
data = data[data[:, 0].argsort()]
data_by_type = [data[data[:,0]==type] for type in unique(data[:,0])]

#compute statistical values
means = [mean(element, axis=0) for element in data_by_type]
maxs = [amax(element, axis=0) for element in data_by_type]
mins = [amin(element, axis=0) for element in data_by_type]

#visualize
f1 = plt.figure(1)
plt.xlabel('pental length')
plt.ylabel('sepal length')
plt.title('Fisher`s Iris Data')
plt.grid(True)
plt.scatter(data_by_type[0][:,2], data_by_type[0][:,4], 20, c="g")
plt.scatter(data_by_type[1][:,2], data_by_type[1][:,4], 20, c="r")
plt.scatter(data_by_type[2][:,2], data_by_type[2][:,4], 20, c="b")
plt.savefig("test4c.png")
#plt.show()

#linear regression
f2 = plt.figure(2)
plt.xlabel('sepal length')
plt.ylabel('class')
plt.title('Fisher`s Iris Data')
plt.grid(True)
plt.scatter(data_by_type[0][:,4], data_by_type[0][:,0], 20, c="g")
plt.scatter(data_by_type[1][:,4], data_by_type[1][:,0], 20, c="r")
plt.savefig("test4d.png")

#LMS Solving
x = concatenate((data_by_type[0][:,4], data_by_type[1][:,4]), axis=0)
y = concatenate((data_by_type[0][:,0], data_by_type[1][:,0]), axis=0)
w = -1 * ones((2, 1))
eta = 1e-4
i = 0
error = sum([(yi - dot([1, xi], w))**2 for xi, yi in zip(x, y)])
diffold = 0

w0 = []
w1 = []
err = []

while error > 1e-5 and i < 1000:
    index = randint(0, x.shape[0]-1)
    xi = array([[1, x[index]]]).T  # x vector of current iteration of shape [1 \n xi]
    diff = y[index] - dot(w.T, xi)  # difference between data value an linear regression value
    dw = eta * diff * xi  # delta weights for convergence
    w += dw
    error = (diff - diffold) ** 2  # sum of squared errors for convergence check
    diffold = diff
    i += 1

    #convergence parameter study
    w0.append(w[0].flatten())
    w1.append(w[1].flatten())
    err.append(error.flatten())


print(i, error)

plt.plot(x, w[0]+w[1]*x)
plt.savefig("test4c.png")

f3 = plt.figure(3)
plt.plot(range(i), err)
f4 = plt.figure(4)
plt.plot(range(i), w0)
plt.plot(range(i), w1)

plt.show()

