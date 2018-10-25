import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.array([5, 7, 15, 28])
y = np.array([50, 79, 124, 300])

# Linear Regression
xm = np.mean(x)
ym = np.mean(y)

xmod = [xi-xm for xi in x]
xmod2 = [(xi-xm)**2 for xi in x]
ymod = [yi-ym for yi in y]

w1 = np.dot(xmod, np.transpose([ymod]))/np.sum(xmod2)
w0 = ym - w1 * xm

ytest = w0+w1*15

#plot
plt.xlabel('Age')
plt.ylabel('Stopping distance [m]')
plt.title('Linear Regression')
plt.grid(True)
plt.plot(x, w0+w1*x)
plt.scatter(x, y, 20, c="g")
plt.scatter(15, w0+w1*15, 20, c="r")
plt.savefig("test.png")
plt.show()