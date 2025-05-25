import numpy as np
from numpy.linalg import inv, solve, matrix_power
import pypoman
from numpy import array, eye, ones, vstack, zeros

#Risoluzione del problema lineare
#min c.Tx s.t. Ax = b, x>=0
#tramite algoritmo di affine scaling

def Affine_scaling(x0,A,b,c,a,toll,maxit):
    it = 0
    n = x0.shape[0]
    e = np.ones(n).reshape(-1,1)

    X0 = np.diag(x0[:,0])
    w = compute_w(x0,A,c)
    r = c - A.T.dot(w)

    while stop_cond(x0,r,toll) == False:
        it += 1
        if it >= maxit:
            break
        X0 = np.diag(x0[:,0])
        w = compute_w(x0,A,c)
        r = c - A.T.dot(w)
        Delta_y = -X0.dot(r)
        if check_unb(Delta_y) == False:
            print("Il problema ha soluzione illimitata")
            break
        if check_opt(Delta_y) == False:
            break
        Theta = (-a/Delta_y[Delta_y<0]).min()
        xk = x0 + Theta*X0.dot(Delta_y)

        if (A.dot(xk) == b).all() == False:
            xk = Feasibile_x0(xk,A,b,a,toll,maxit)

        x0 = xk

    optimal = c.T.dot(xk)
    return xk, it, optimal[0,0]

def compute_w(x0,A,c):
    X0 = np.diag(x0[:,0])
    D = matrix_power(X0, 2)
    v = A.dot(D).dot(c)
    w = solve(A.dot(D).dot(A.T), v)
    return w

def check_unb(Delta):
    return any(Delta < 0)

def check_opt(Delta):
    return any(Delta != 0)

def stop_cond(x0,r,toll):
    X0 = np.diag(x0[:,0])
    n = x0.shape[0]
    e = np.ones(n).reshape(-1,1)
    return  any(r < 0) == False and e.T.dot(X0).dot(r) <= toll

#############################################

def Feasibile_x0(x,A,b,a,toll,maxit):
    it = 0
    n = x.shape[0]
    rho = b - A.dot(x)
    A_cap = np.append(A, rho, axis=1)
    c = np.array([0 for i in range(n)] + [1]).reshape(-1,1)
    x0 = np.concatenate((x,[[1]]))

    X0 = np.diag(x0[:,0])
    w = compute_w(x0,A_cap,c)
    r = c - A_cap.T.dot(w)

    while stop_cond(x0,r,toll) == False:
        it += 1
        if it >= maxit:
            break
        X0 = np.diag(x0[:,0])
        X0_ = np.diag(x0[:,0][:-1])
        D_ = matrix_power(X0_, 2)
        w = compute_w(x0,A_cap,c)
        r = c - A_cap.T.dot(w)

        v = solve(A.dot(D_).dot(A.T), rho)
        Delta_x = D_.dot(A.T).dot(v)
        Delta_x = np.concatenate((Delta_x,[[-1]]))
        Theta = a*(-x0[Delta_x<0]/Delta_x[Delta_x<0]).min()
        xk = x0 + Theta*Delta_x
        x0 = xk
    x_feas = xk[:-1]
    return x_feas

#################################################

A = np.array([[1,-1,1,0],[0,1,0,1]])
b = np.array([15,15]).reshape(-1,1)
c = np.array([-2,1,0,0]).reshape(-1,1)
a = 0.99
toll = 10**-12
maxit = 50
x = np.ones(4).reshape(-1,1)
x0 = Feasibile_x0(x,A,b,a,toll,maxit)
print(x0)
#x0 = np.array([10,2,7,13]).reshape(-1,1)
optimal, it, xk = Affine_scaling(x0,A,b,c,a,toll,maxit)
print(xk,it,optimal)

################################################
