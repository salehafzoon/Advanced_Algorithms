from pprint import pprint
import random as rn
import math
import copy
from random import randrange
import matplotlib.pyplot as plt
import time
import openpyxl
import os

import numpy as np
from collections import defaultdict

ITERATIONS = 100
SWARM_SIZE = 40

DEBUG = False
TIMER_MODE = False
EXEl_WRITE = False

Rosenbrock = "Rosenbrock"
Step = "Step"
Ackley = "Ackley"
Griewank = "Griewank"
Rastrigin = "Rastrigin"
Genr_Penalized = "Generalized Penalized"

FUNCTIONS = [(Rosenbrock, 30), (Step, 100), (Ackley, 32),
             (Griewank, 600), (Rastrigin, 512), (Genr_Penalized, 50)]

TEST_FUNC = None
BOUND = None
Ns = [10, 30, 50]


class Particle(object):
    def __init__(self, n):
        self.n = n
        self.x = []
        self.v = []
        self.f = None
        self.pbest = None
        self.bound = None
        self.positioning()

    def positioning(self):
        for i in self.n:
            self.x[i] = rn.uniform(-BOUND, BOUND)
            self.v[i] = 0.1 * self.x[i]
            self.calculate_f()

    def calculate_f(self):
        fVal = 0
        if TEST_FUNC == Rosenbrock:
            for i in range(len(self.n)-1):
                fVal += (100 * (x[i + 1] - x[i] ** 2) ** 2) + ((x[i] - 1) ** 2)

        elif TEST_FUNC == Step:
            pass

        self.f = fVal
        

    def updatePbest(self):
        self.pbest = min(self.pbest,self.f)

    def updateVelocityAndPos(self):
        for i in range(self.n):
            c1 = rn.uniform(0,2)
            c2 = rn.uniform(0,2)
            
            v[i] = w[i] * v[i] + c
    
    def __str__(self):
        return "x: " + str(self.x[:5]) + "f(x): " + str(self.f)

    def __repr__(self):
        return str(self)


class Swarm(object):

    def __init__(self, size, n):
        self.size = size
        self.n = n
        self.gbest = None
        self.particles = []

    def initial(self):
        for _ in range(self.size):
            self.particles.append(Particle(n))
    
    def updateGbest(self):
        for par in self.particles:
            self.gbest = min(self.pbest,par.pbest)


    def __str__(self):
        return str(self.number) + str(self.getCord())

    def __repr__(self):
        return str(self)


def printResult(answers):

    minAns = min(answers, key=lambda t: t[1])
    maxAns = max(answers, key=lambda t: t[1])
    variance = round(math.sqrt(np.var([ans[1]for ans in answers])), 3)

    print("\nbest[0:10]=", minAns[0][0:10], "\tmin cost:", minAns[1])
    if TIMER_MODE == False:
        print("worst[0:10]=", maxAns[0][0:10], "\tmax cost:",
              max(answers, key=lambda t: t[1])[1])
        print("\naverage cost:", sum(ans[1] for ans in answers)/len(answers))
        print("\nvariance of costs:", variance)

        print("\nmin time:", min(answers, key=lambda t: t[2])[2])
        print("avg time:", str(sum(float(ans[2])
                                   for ans in answers)/len(answers))[0:6])
        print("max time:", max(answers, key=lambda t: t[2])[2])


def writeResultToExel(file_name, answers, myRow):
    minCost = min(answers, key=lambda t: t[1])[1]
    if TIMER_MODE == False:
        maxCost = max(answers, key=lambda t: t[1])[1]
        avgCost = sum(ans[1] for ans in answers)/len(answers)
        costVariance = round(math.sqrt(np.var([ans[1]for ans in answers])), 3)

        avgTime = str(sum(float(ans[2])for ans in answers)/len(answers))[0:6]

    wbkName = 'one_min_Results.xlsx'
    wbk = openpyxl.load_workbook(wbkName)
    for wks in wbk.worksheets:
        myCol = 7

        # wks.cell(row=myRow, column=1).value = file_name

        wks.cell(row=myRow, column=myCol).value = minCost
        if TIMER_MODE == False:
            wks.cell(row=myRow, column=myCol+1).value = avgCost
            wks.cell(row=myRow, column=myCol+2).value = maxCost
            wks.cell(row=myRow, column=myCol+3).value = costVariance

            wks.cell(row=myRow, column=myCol+4).value = avgTime

    wbk.save(wbkName)
    wbk.close


def plotResult(generation, bests, averages):

    plt.plot(list(range(generation)), bests, '-', color="gray",
             label='best of generations', linewidth=1.52)

    plt.xlabel('best solution')
    plt.ylabel('generation')

    plt.show()

    plt.plot(list(range(generation)), averages, 'bo-',
             label='avg of generations', linewidth=1.5)

    plt.xlabel('best solution')
    plt.ylabel('generation')

    plt.show()


def plotProgress(bests, avgs):

    plt.plot(list(range(len(bests))), bests, '-', color="gray",
             label='best of generations', linewidth=1.5)

    plt.xlabel('best solution')
    plt.ylabel('generation')

    plt.show()

    plt.plot(list(range(len(avgs))), avgs, 'bo-',
             label='avg of generations', linewidth=1.5)

    plt.xlabel('avg solution')
    plt.ylabel('generation')

    plt.show()


def PSO(SWARM_SIZE, N, ITERATIONS=50, DEBUG=True):

    timer = time.time()
    if TIMER_MODE:
        ITERATIONS = 10000000000

    # particles random initial 
    swarm = Swarm(SWARM_SIZE, N)
    
    for i in range(ITERATIONS):

        for particle in swarm.particles:
            particle.updatePbest()
        
        swarm.updateGbest()

        for particle in swarm.particles:
            particle.updateVelocityAndPos()

if __name__ == '__main__':

    for test_func in FUNCTIONS:

        (TEST_FUNC, BOUND) = test_func

        for N in Ns:
            PSO(SWARM_SIZE, N, ITERATIONS, DEBUG)