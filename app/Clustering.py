"""
This code is the implementation of K-Means method to find the amount of clusters
that are needed for grouping a data set.
This methods belong to unsupervised learning in ML.
The purpose of this code is to implement the method itself, step by step, rather
than using a module.
"""
from random import uniform


def initializeCentroids(nc, x, y):
    centroids = []
    for i in range(nc):
        ind = int(uniform(0, len(x)))
        newCentroid = [x[ind], y[ind]]
        if newCentroid not in centroids:
            centroids.append(newCentroid)
    return centroids


def calculateNewCentroids(kgc):
    auxCentroids = []
    for groupN in kgc:
        xc = 0;
        yc = 0
        for i in range(len(groupN)):
            p = groupN[i]
            xc += p[0];
            yc += p[1]
        if len(groupN) > 0:
            xc = xc / (len(groupN));
            yc = yc / (len(groupN))
        auxNewCentroid = [xc, yc]
        auxCentroids.append(auxNewCentroid)
    return auxCentroids


def calculateInertia(lbl, kc, x, y):
    distance = 0
    for i in range(len(x)):
        _label = lbl[i]
        kcp = kc[_label]
        distance += ((x[i] - kcp[0]) ** 2 + (y[i] - kcp[1]) ** 2)
    return distance


def kmean(k, ic, ag):
    kcentroids = initializeCentroids(k, ic, ag)
    controller = True
    while controller:
        label = []

        for i in range(len(ic)):
            samplePoint = [ic[i], ag[i]]

            aux = kcentroids[0]
            lowerDistance = ((samplePoint[0] - aux[0]) ** 2 + (samplePoint[1] - aux[1]) ** 2) ** (1 / 2)
            lowerDistanceInd = 0

            for j, ind in zip(kcentroids, range(len(kcentroids))):
                dis = ((samplePoint[0] - j[0]) ** 2 + (samplePoint[1] - j[1]) ** 2) ** (1 / 2)
                if dis < lowerDistance:
                    lowerDistance = dis
                    lowerDistanceInd = ind
            label.append(lowerDistanceInd)
        kgroups = []
        for i in range(k):
            group = []
            for j in range(len(ic)):
                if label[j] == i:
                    nPoint = [ic[j], ag[j]]
                    group.append(nPoint)
            kgroups.append(group)

        ###Calculate new centroids
        newCentroids = calculateNewCentroids(kgroups)
        if newCentroids == kcentroids:
            controller = False
        else:
            kcentroids = newCentroids
    inertia = calculateInertia(label, kcentroids, ic, ag)
    return inertia, label
