# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 12:11:06 2021

@author: fassi
"""

import matplotlib.pyplot as plt
import math
from pyomo.environ import *

def ParseFile(filename):
    doc = open(filename, 'r')
    # Salta le prime 3 righe
    for _ in range(3):
        doc.readline()
    # Leggi la dimensione
    n = int(doc.readline().split(' ')[2])
    # sala riga
    doc.readline()
    # Leggi la capacita
    C = int(doc.readline().split(' ')[2])
    # sala riga
    doc.readline()
    # Leggi posizioni
    Ps = {}
    for row in doc:
        row = row.rstrip().split(' ')
        if row[0] == 'DEMAND_SECTION':
            break
        row = list(map(lambda z: int(z), row))
        Ps[row[0]] = (row[1], row[2])
    # Leggi posizioni
    Ds = {}
    for row in doc:
        row = row.rstrip().split(' ')
        if row[0] == 'DEPOT_SECTION':
            break
        row = list(map(lambda z: int(z), row))
        Ds[row[0]] = row[1]
    d = int(next(doc).rstrip())

    return n, C, Ps, Ds, d


def Distance(A, B):
    return int(round(math.sqrt((A[0]-B[0])**2 + (A[1] - B[1])**2),0))

    
def DisegnaSegmento(A, B, ax):
    """ 
    Disegna un segmento nel piano dal punto A a al punto B
    Vedi manuale a: http://matplotlib.org/api/pyplot_api.html
    """
    # Disegna il segmento
    ax.plot([A[0], B[0]], [A[1], B[1]], 'b', lw=0.75)
    # Disegna gli estremi del segmento
    DisegnaPunto(A, ax)
    DisegnaPunto(B, ax)
    

def DisegnaPunto(A, ax):
    """
    Disegna un punto nel piano
    """
    ax.plot([A[0]], [A[1]], 'bo', alpha=0.5)
    
def PlotSolution(Xs, Ws, Es):
    fig, ax = plt.subplots()
    for i,j in Es:
        DisegnaSegmento(Ps[i], Ps[j], ax)
    
    ax.scatter([i for i,j in Xs[1:]], [j for i,j in Xs[1:]], 
                s=Ws[1:], alpha=0.3, cmap='viridis')
    
    for i in range(len(Xs[1:])):
        ax.annotate(str(i+1), Xs[i])
    
    plt.plot([Xs[0][0]], [Xs[0][1]], marker='s', color='red', alpha=0.5)
    plt.axis('square')
    plt.axis('off')

def VRP(n, K, C, Ps, Ds, d, F, TimeLimit):
    m = ConcreteModel()
    
    m.I = RangeSet(len(Ps))
    m.J = RangeSet(len(Ps))
    
    m.x = Var(m.I, m.J, domain = Binary)
    
    Es = []
    
    for i in m.I:
        for j in m.J:
            if i !=j:
                Es.append( (i,j) )
    
    m.obj = Objective(expr = sum(F(Ps[i], Ps[j])*m.x[i,j] for i,j in Es))
    
    #Vincoli archi uscenti
    m.outdegree = ConstraintList()
    for i in m.I:
        if i != d:
            Ls = list(filter(lambda z: z!=i, Ps))
            m.outdegree.add(expr = sum(m.x[i,j] for j in Ls) == 1)
    #Vincoli archi entarnti
    m.indegree = ConstraintList()
    for j in m.J:
        if j != d:
            Ls = list(filter(lambda z: z!=j, Ps))
            m.indegree.add(expr = sum(m.x[i,j] for i in Ls) == 1)
            
    Ls = list(filter(lambda z: z!=d, Ps))
    m.d1 = Constraint( expr = sum(m.x[d,i] for i in Ls) >= 3)
    m.d1 = Constraint( expr = sum(m.x[i,d] for i in Ls) >= 3)        
    
    m.pairs = ConstraintList()
    
    
    for a,b in [(19,20), (15,18), (7,2), (5,6), (22,8)]:
        Fs = []
        for i in Ps:
            if i == a or i == b:
                for j in Ps:
                    if i!= j and j!= a and j!= b:
                        Fs.append((i,j))
        m.pairs.add(expr = sum(m.x[i,j] for i,j in Fs) >= 1)
    
    
    for a,b,c in [(5,6,9)]:
        Fs = []
        for i in Ps:
            if i == a or i == b or i == c:
                for j in Ps:
                    if i!= j and j!= a and j!= b and j!= c:
                        Fs.append((i,j))
        m.pairs.add(expr = sum(m.x[i,j] for i,j in Fs) >= 1)
    
    for a,b,c in [(10,11,14)]:
        Fs = []
        for i in Ps:
            if i == a or i == b or i == c:
                for j in Ps:
                    if i!= j and j!= a and j!= b and j!= c:
                        Fs.append((i,j))
        m.pairs.add(expr = sum(m.x[i,j] for i,j in Fs) >= 2)
    
    
    #Fisssiamo un sottoinsieme S di nodi
    
    #Troviamo tutte le coppie (i,j) con i in S, j non in S (che abbiamo chiamato Fs)
    #Poniamo il vincolo:
        #\sum_(ij in Fs) x_ij >= stima per difetto del numero di veicoli che servono
        #per servire tutti i nodi in S
        #Ovvero la ceil(sommatoria delle domande in S/ capacità)
    
    
    #risolviamo il modello
    
    sol = SolverFactory('glpk').solve(m, tee = True)
    
    
    #Controllo lo status!
    sol_json = sol.json_repn()
    
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    
    selected = []
    for i in m.I:
        for j in m.J:
            if i!= j:
                if m.x[i,j]() > 0:
                    selected.append( (i,j) )
    
    return m.obj(), selected

n, C, Ps, Ds, d = ParseFile("cvrp_data.txt")
# fobj, Es = cMTZ(n, 3, C, Ps, Ds, d, Distance, 60) 
print(Ps)  

Pool=[]
# fobj, Es, Pool = VRPCut(n, 5, C, Ps, Ds, d, Distance, 60)   
fobj, Es = VRP(n, 4, C, Ps, Ds, d, Distance, 60) 

print('valore funzione obiettivo:', fobj)

Xs = [p for p in Ps.values()]
Ws = [w for w in Ds.values()]

PlotSolution(Xs, Ws, Es)














