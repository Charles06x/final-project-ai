import logisticRegression
import Clustering
from poliTransformation import modifyDataSet
import numpy as np
import matplotlib.pyplot as plt
import easygui
from copy import copy

dataSet = np.loadtxt("../docs/cluster1.txt", delimiter=';')
incomes = dataSet[:, 0]
ages = dataSet[:, 1]

xplt, yplt = [], []
ini = 1
for i in range(ini, 11):
    km, l = Clustering.kmean(i, incomes, ages)
    xplt.append(i)
    yplt.append(km)

# #Recta pa
points = [(xplt[0], yplt[0]), (xplt[-1], yplt[-1])]

x_coords, y_coords = zip(*points)

A = np.vstack([x_coords, np.ones(len(x_coords))]).T

m, c = np.linalg.lstsq(A, y_coords)[0]

nx, ny, nny = [], [], []
mayor = 0
for i in range(ini, 11):
    nx.append(i)
    ny.append(m * i + c)
    nny.append(m * i + c - yplt[i - ini])
    auxy = m * i + c

    d = ((i - xplt[i - ini]) ** 2 + ((m * i + c) - yplt[i - ini]) ** 2) ** (1 / 2)
    if d > mayor:
        mayor = d
        ncluster = i

print("Numero de clusters optimo: ", ncluster)
inert, lb = Clustering.kmean(ncluster, incomes, ages)

plt.subplot(211)
plt.xlabel("# Clusters")
plt.ylabel("Inercia")
plt.plot(xplt, yplt, "bo-", label="Inercia vs # Clusters")
plt.plot(nx, ny, "ro-", label="Linea recta auxiliar")
plt.subplot(212)
plt.scatter(dataSet[:, 0], dataSet[:, 1], c=lb)
plt.show()

myvar = easygui.enterbox("¿Es realmente " + str(ncluster) + " el número de clusters optimo?\n Si no, ingrese el número óptimo:")
if myvar != '' and int(myvar) > 2:
    ncluster = int(myvar)

    print("Numero de clusters optimo: ", ncluster)
    inert, lb = Clustering.kmean(ncluster, incomes, ages)
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=lb)
    plt.show()

print("\n################################################")
print("##########     Regresion Logistica     #########")
print("################################################")

# Load Dataset
dataSet = np.loadtxt("../docs/datasetRegLog1.txt", delimiter=';')
x = dataSet[:, [0, 1]]
y = dataSet[:, 2]

print("\n\n\n\t\tLineal Model\n")

thetas = logisticRegression.logistic_regression(x, y)
print(thetas)
logisticRegression.testing(x, y, thetas)

print("\n\n\n\t\tPoly Model\n")

i = easygui.enterbox("¿De qué grado desea el polinomio para la regresión logística?")
if i != '' and int(i) > 1:
    _x = modifyDataSet(x, int(i))
    thetas = logisticRegression.logistic_regression(_x, y)
    print(thetas)
    logisticRegression.testing(_x, y, thetas)




