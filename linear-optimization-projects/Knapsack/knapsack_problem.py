# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:34:38 2021

@author: fassi
"""

from pyomo.environ import *

I = range(5) #oggetti
C = [2,3,1,4,3] #vettore dei valori
A = [2,4,2,1,6] #vettore dei pesi
B = 9 #budget o il peso da non sforare

model = ConcreteModel()

model.x = Var(I, within = Binary)

model.obj = Objective(expr = sum(C[i]*model.x[i] for i in I), sense = maximize)

model.capacity = Constraint(expr = sum(A[i]*model.x[i] for i in I) <= B)

# Solve the model
sol = SolverFactory('glpk').solve(model)

# Basic info about the solution process
for info in sol['Solver']:
    print(info)
    
# Report solution value
print("Optimal solution value: z =", model.obj())
print("Decision variables:")
for i in I:
    print("x_{} = {}".format(i, model.x[i]()))
print("Capacity left in the knapsack:", B-model.capacity())