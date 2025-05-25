import numpy as np
from numpy.linalg import inv, solve, matrix_power, norm
from numpy import array, eye, ones, vstack, zeros


def Affine_scaling(x0,A,b,c,a,toll,maxit):
    optimal_list = []
    points_list = []
    Theta_list = []

    it = 0
    n = x0.shape[0]

    for it in range(maxit):
        X0 = np.diag(x0[:,0])
        w = compute_w(x0,A,c)
        r = c - A.T.dot(w)
        if dual_feas(c,r,toll) == True and slackness(x0,c,b,w) == True:
            xk = x0
            break
        Delta_y = -X0.dot(r)
        if any(Delta_y <= 0) == False:
            print("Il problema ha soluzione illimitata")
            break
        if any(Delta_y != 0) == False:
            xk = x0
            break
        Theta = (-a/Delta_y[Delta_y<0]).min()
        xk = x0 + Theta*X0.dot(Delta_y)

        if primal_feas(x,A,b,toll) == False:
            xk = Feasibile_x0(xk,A,b,a,toll,maxit)

        x0 = xk

        optimal = c.T.dot(xk)

        optimal_list.append(optimal[0,0])
        points_list.append(x0)
        Theta_list.append(Theta)

    return xk, it, optimal[0,0],optimal_list,points_list,Theta_list


def compute_w(x0,A,c):
    X0 = np.diag(x0[:,0])
    D = matrix_power(X0, 2)
    v = A.dot(D).dot(c)
    w = solve(A.dot(D).dot(A.T), v)
    return w

def dual_feas(c,r,toll):
    num = norm(r[r<0])
    den = norm(c[r<0]) + 1
    return  num/den <= toll

def primal_feas(x,A,b,toll):
    num = norm(A.dot(x) - b)
    den = norm(b) + 1
    return num/den <= toll

def slackness(x,c,b,w):
    return c.T.dot(x) - b.T.dot(w) <= 0


def Feasibile_x0(x0,A,b,a,toll,maxit):
    n = x0.shape[0]
    v = b - A.dot(x0)
    A_cap = np.append(A, v, axis=1)
    c_cap = np.array([0 for i in range(n)] + [1]).reshape(-1,1)
    x0_cap = np.concatenate((x0,[[1]]))

    for it in range(maxit):
        X0 = np.diag(x0[:,0])
        X0_cap = np.diag(x0_cap[:,0])
        D = matrix_power(X0, 2)
        w = compute_w(x0_cap,A_cap,c_cap)
        r = c_cap - A_cap.T.dot(w)

        if dual_feas(c_cap,r,toll) == True and slackness(x0_cap,c_cap,b,w) == True:
            break
        Delta = D.dot(A.T).dot(solve(A.dot(D).dot(A.T),v))
        Delta = np.concatenate((Delta,[[-1]]))
        if any(Delta <= 0) == False:
            print("Il problema ha soluzione illimitata")
            break
        if any(Delta != 0) == False:
            break
        Theta = a*(-x0_cap[Delta<0]/Delta[Delta<0]).min()
        xk_cap = x0_cap + Theta*Delta
        x0_cap = xk_cap
        x0 = x0_cap[:-1]
    x_feas = x0
    return x_feas


#####################################

A = np.array([[1,-1,1,0],[0,1,0,1]])
b = np.array([15,15]).reshape(-1,1)
c = np.array([-2,1,0,0]).reshape(-1,1)
a = 0.99
toll = 10**-18
maxit = 50
x = np.ones(4).reshape(-1,1)
x0 = Feasibile_x0(x,A,b,a,toll,maxit)
# x0 = np.array([10,2,7,13]).reshape(-1,1)
print("punto iniziale:",x0)
xk, it, optimal,optimal_list,points_list,Theta_list = Affine_scaling(x0,A,b,c,a,toll,maxit)
print(xk,it,optimal)

####################################


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

y = np.array([[0,0], [0,15], [30,15], [15,0]])

p = Polygon(y, facecolor = 'aquamarine')

fig,ax = plt.subplots()

ax.add_patch(p)
ax.set_xlim([0,35])
ax.set_ylim([0,20])

l = []
i = 0
plt.scatter(x0[:-2][0],x0[:-2][1],marker='o',c='r',zorder=2)
for point in points_list:
    l.append(point[:-2].reshape(1,2).tolist()[0])
    plt.scatter(point[0],point[1],marker='o',c='r',zorder=2)

plt.plot([x0[:-2][0]] + [l[i][0] for i in range(len(l))], [x0[:-2][1]] + [l[i][1] for i in range(len(l))])

plt.show()

############################

from tabulate import tabulate

D = {'Valore ottimo': optimal_list, 'Theta': Theta_list, 'Punto': points_list}
print(tabulate(D, headers='keys'))
