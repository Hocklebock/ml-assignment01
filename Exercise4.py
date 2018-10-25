import matplotlib.pyplot as plt
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

while error > 1e-3 and i < 1000:
    error = 0
    # iterate over all data values
    for index in range(x.shape[0]):
        xi = array([[1, x[index]]]).T #x vector of current iteration of shape [1 \n xi]
        diff = y[index] - dot(w.T, xi) #difference between data value an linear regression value
        dw = eta * diff * xi # delta weights for convergence
        w += dw
        error += diff**2 # sum of squared errors for convergence check
    i += 1

print(i, sum([(yi - dot([1, xi], w))**2 for xi, yi in zip(x, y)]))

plt.plot(x, w[0]+w[1]*x)
plt.savefig("test4c.png")
plt.show()

