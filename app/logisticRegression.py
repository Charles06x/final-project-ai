# -*- coding: utf-8 -*-
import numpy as np


def logistic_regression(_x, _y):
    _thetas = [0] * (len(_x[0]) + 1);
    alpha = 0.0009
    for n in range(0, 1000):
        aux = [0] * len(_thetas);
        sum = 0

        for j in range(len(_thetas)):
            for i in range(int(len(_x) * 0.7)):
                h = 0
                for l in range(len(_thetas)):   # theta.T * x
                    if l == 0:
                        h = _thetas[l]
                    else:
                        h += _thetas[l] * _x[i][l-1]
                e = 1 / (1 + np.exp((-h)))      #Apply sigmoide
                if j == 0:
                    sum += (e - _y[i])  # Xo = 1
                else:
                    sum += (e - _y[i]) * (_x[i][j - 1])

            aux[j] = _thetas[j] - (alpha / (int(len(_x) * 0.7))) * sum
        _thetas = aux
    return _thetas


# ###Test the resultant formula


def testing(_x, _y, _thetas):
    dsq = 0
    _tp, _tn, _fp, _fn = 0, 0, 0, 0  # Confusion matrix
    for n in range(int(len(_x) * 0.7), len(_x)):
        for l in range(len(_thetas)):
            if l == 0:
                h = _thetas[0]
            h += _thetas[l] * _x[n][l - 1]

        g = 1 / (1 + np.exp(-h))
        if g < 0.5:  # If predicted is negative...
            if _y[n] == 0:  # And actual is negative
                _tn += 1
            if _y[n] == 1:  # and actual is positive
                _fn += 1
        else:  # If predicted is positive...
            if _y[n] == 1:  # And actual is positive
                _tp += 1
            if _y[n] == 0:  # And Actual is negative
                _fp += 1
        dsq += (g - _y[n]) ** 2

    error = (dsq / (len(_x) - int(len(_x) * 0.7)))
    print("Error: ", error)

    # #### Precision, Recall and F1-Score
    # # Precision:
    precision = _tp / (_tp + _fp)
    # #Recall:
    recall = _tp / (_tp + _fn)
    # #F1-Score
    f_score = (2 * precision * recall) / (precision + recall)

    print("#########################################")
    print("#########################################")
    print("\t   CONFUSION MATRIX")
    print("         Negative     Positive")
    print("Negative   {0}           {1}".format(_tn, _fp))
    print("Positive   {0}           {1}".format(_fn, _tp))
    print("#########################################")
    print("#########################################")

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f_score: ", f_score)


# ##Charles Acevedo
