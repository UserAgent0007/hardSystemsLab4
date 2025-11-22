import math as m
import numpy as np
import pandas as pd

def g_1(h, y_n, y_nPlus1):

    return (y_nPlus1 - y_n )/ h

def g_2(h, y_nMinus1, y_n, y_nPlus1):

    return (3 * y_nPlus1 - 4 * y_n + y_nMinus1) / (2 * h)

def g_3(h, y_nMinus2, y_nMinus1, y_n, y_nPlus1):

    return (1 / h) * (11/6 * y_nPlus1 - 3 * y_n + 1.5 * y_nMinus1 - 1/3 * y_nMinus2)

def f_derivative_g1(alpha, h, y_n, y_nPlus1):
    g_val = g_1(h, y_n, y_nPlus1)
    return (-alpha / h) * (1 - 0.5 * m.cos(0.5 * g_val))

def f_derivative_g2(alpha, h, y_nMinus1, y_n, y_nPlus1):
    g_val = g_2(h, y_nMinus1, y_n, y_nPlus1)
    return (-alpha / h) * (1 - 0.5 * m.cos(0.5 * g_val))

def f_derivative_g3(alpha, h, y_nMinus2, y_nMinus1, y_n, y_nPlus1):
    g_val = g_3(h, y_nMinus2, y_nMinus1, y_n, y_nPlus1)
    return (-alpha / h) * (1 - 0.5 * m.cos(0.5 * g_val))

def f1(h, y_n, y_nPlus1, x):
    g = g_1(h, y_n, y_nPlus1)
    return g - m.sin(0.5 * g) - 2 * x + m.sin(x)

def f2(h, y_nMinus1, y_n, y_nPlus1, x):
    g = g_2(h, y_nMinus1, y_n, y_nPlus1)
    return g - m.sin(0.5 * g) - 2 * x + m.sin(x)

def f3(h, y_nMinus2, y_nMinus1, y_n, y_nPlus1, x):
    g = g_3(h, y_nMinus2, y_nMinus1, y_n, y_nPlus1)
    return g - m.sin(0.5 * g) - 2 * x + m.sin(x)


def newthoMethod (h, x_next,  *args):

    elements = list(args)
    # y_0 = elements[0]
    y_nabl = 0

    if (len(elements) == 1):

        y_nabl = elements[0]
        flag = True
        iteration = 1

        while flag and iteration < 1000:

            y_nabl_next = y_nabl - f1(h, elements[0], y_nabl, x_next) / f_derivative_g1(-1, h, elements[0], y_nabl)

            if m.fabs(y_nabl_next - y_nabl) <= 0.0005:

                flag = False

            y_nabl = y_nabl_next

            iteration += 1

    elif len(elements) == 2:

        y_nabl = elements[1]
        flag = True
        iteration = 1

        while flag and iteration < 1000:

            y_nabl_next = (y_nabl - f2(h, elements[0], elements[1], y_nabl, x_next) /
                           f_derivative_g2(-3/2, h, elements[0], elements[1], y_nabl))

            if m.fabs(y_nabl_next - y_nabl) <= 0.0005:
                flag = False

            y_nabl = y_nabl_next

            iteration += 1

    elif len(elements) == 3:

        y_nabl = elements[2]
        flag = True
        iteration = 1

        while flag and iteration < 1000:

            y_nabl_next = (y_nabl - f3(h, elements[0], elements[1], elements[2], y_nabl, x_next) /
                           f_derivative_g3(-11/6, h, elements[0], elements[1], elements[2], y_nabl))

            if m.fabs(y_nabl_next - y_nabl) <= 0.0005:
                flag = False

            y_nabl = y_nabl_next

            iteration += 1

    return y_nabl

y = [0]
x = [0]
h = 0.01

y.append(newthoMethod(h, x[-1] + h, y[0]))
x.append(x[-1] + h)

y.append(newthoMethod(h, x[-1] + h, y[0], y[1]))
x.append(x[-1] + h)

while x[-1] < 1:

    y.append(newthoMethod(h, x[-1] + h, y[-3], y[-2], y[-1]))
    x.append(x[-1] + h)

x = np.reshape(np.array(x), (-1, 1))
y_real = np.reshape(np.array(x)**2, (-1,1))
y = np.reshape(np.array(y), (-1,1))
delta_y = np.reshape(np.fabs(y - y_real), (-1,1))

result = pd.DataFrame(np.hstack((x, y, y_real, delta_y)), columns=['x', 'y_approx', 'y_real', 'delta_y'])

print (result)