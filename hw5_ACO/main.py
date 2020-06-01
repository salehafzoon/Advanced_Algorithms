from pprint import pprint
import random as rn
import math
import copy
from random import randrange
import matplotlib.pyplot as plt
import time
import openpyxl
import os
import tsp

from tspy2 import TSP
from tspy2.solvers import NN_solver
from tspy2.solvers import TwoOpt_solver
import numpy as np
from collections import defaultdict

DEBUG = False
ANTS_NUM = 10
iterations = 500
PheromoneConstant = 1.0
DecayConstant = 0.2
ALPHA = 1
Beta = 2   # Heuristic constant
RHO = 0.9
Q0 = 0.9  # probability parameter

bests = []
averages = []

TIMER_MODE = False
EXEl_WRITE = False


def euclideanDist_2D(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class Node(object):

    def __init__(self, number, x, y):
        self.number = number
        self.x = x
        self.y = y
        self.demand = None
        self.cluster = None

    def getCord(self):
        return (self.x, self.y)

    def __str__(self):
        return str(self.number) + str(self.getCord())

    def __repr__(self):
        return str(self)


class Problem(object):

    def __init__(self):
        self.name = None
        self.dimention = None
        self.capacity = None
        self.depotCluster = None
        self.clusters = []
        self.minDistances = defaultdict(dict)

    # calculating min distance of each per of clusters
    def calculateMinDists(self):

        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):

                # distance not calculated
                if not(j in self.minDistances[i]):
                    cl1 = self.clusters[i]
                    cl2 = self.clusters[j]

                    minDist = 1000000000000
                    for n1 in cl1:
                        for n2 in cl2:
                            dist = euclideanDist_2D(n1, n2)
                            if minDist > dist:
                                minDist = dist

                    self.minDistances[i][j] = minDist

        # print(self.minDistances)

    def __str__(self):
        return 'depot cluster: ' + str(self.depotCluster)

    def __repr__(self):
        return str(self)


class Ant(object):

    def __init__(self):
        self.solution = []

    def nextMove(self, problem):

        # finding nearest cluster to move
        pass

    def __str__(self):
        return 'solution:' + str(self.solution)

    def __repr__(self):
        return str(self)


class Colony(object):

    def __init__(self, problem):
        self.ants = []
        self.best = None
        self.problem = problem
        self.T = self.initialTrail()
        self.ants = [Ant() for _ in range(ANTS_NUM)]

    def initialTrail(self):
        size = len(self.problem.clusters)
        self.T = [[1 for _ in range(size)] for _ in range(size)]

    def antPositioning(self):

        # remove depot cluster form cluster list
        clusters = [c for c in list(
            self.problem.clusters.keys()) if c != self.problem.depotCluster]

        # randomly position m ants on n clusters
        for ant in self.ants:
            cluster = rn.choice(clusters)
            ant.solution = [cluster]

    def pheromoneUpdate(self, bestSolution):
        pass

    def __str__(self):
        return str(best) + 'as best answer by ' + str(len(ants)) + 'ants'

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

    print("\naverage vehicels:", sum(ans[3] for ans in answers)/len(answers))


def writeResultToExel(file_name, answers, myRow):
    minCost = min(answers, key=lambda t: t[1])[1]
    if TIMER_MODE == False:
        maxCost = max(answers, key=lambda t: t[1])[1]
        avgCost = sum(ans[1] for ans in answers)/len(answers)
        costVariance = round(math.sqrt(np.var([ans[1]for ans in answers])), 3)

        minTime = min(answers, key=lambda t: t[2])[2]
        maxTime = max(answers, key=lambda t: t[2])[2]
        avgTime = str(sum(float(ans[2])for ans in answers)/len(answers))[0:6]

    avgVehc = sum(ans[3] for ans in answers)/len(answers)

    wbkName = 'Results.xlsx'
    wbk = openpyxl.load_workbook(wbkName)
    for wks in wbk.worksheets:
        myCol = 7

        # wks.cell(row=myRow, column=1).value = file_name

        wks.cell(row=myRow, column=myCol).value = minCost
        if TIMER_MODE == False:
            wks.cell(row=myRow, column=myCol+1).value = avgCost
            wks.cell(row=myRow, column=myCol+2).value = maxCost
            wks.cell(row=myRow, column=myCol+3).value = costVariance

            wks.cell(row=myRow, column=myCol+4).value = minTime
            wks.cell(row=myRow, column=myCol+5).value = avgTime
            wks.cell(row=myRow, column=myCol+6).value = maxTime

        wks.cell(row=myRow, column=myCol+7).value = avgVehc

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


def loadInstance(file):
    problem = Problem()
    with open(file) as f:
        lines = [line.rstrip() for line in f]

        problem.name = lines[0].split(':')[1]
        problem.dimention = int(lines[3].split(':')[1])
        problem.capacity = int(lines[4].split(':')[1])

        problem.clusters = {}

        # NODE_COORD_SECTION first index
        cordSecIndex = 8
        # DEMAND_SECTION first index
        demSecIndex = cordSecIndex + problem.dimention + 1
        # CLUSTER_SECTION first index
        cludSecIndex = demSecIndex + problem.dimention + 1

        # Golden type instance . its cluster number start from 1
        if problem.name == '':
            x = -1
        else:
            x = 0

        problem.depotCluster = int(lines[cludSecIndex].split()[1]) + x

        for i in range(problem.dimention):
            data = lines[cordSecIndex+i].split()
            node = Node(int(data[0]), float(data[1]), float(data[2]))

            node.demand = int(lines[demSecIndex+i].split()[1])
            node.cluster = int(lines[cludSecIndex+i].split()[1]) + x
            problem.clusters.setdefault(node.cluster, []).append(node)

    problem.calculateMinDists()
    return problem


def ACS(problem, iterations=50, debug=True):

    timer = time.time()
    if TIMER_MODE:
        maxGeneration = 100000000

    # number of problem clusters
    size = len(problem.clusters)

    colony = Colony(problem)
    for _ in range(iterations):

        colony.antPositioning()
        for _ in range(size):
            for ant in colony.ants:
                ant.nextMove(problem)

        if DEBUG:
            print("best:",)

    return colony.best


if __name__ == '__main__':

    myRow = 3

    problem = loadInstance('instances/Marc/a-n14-c4.ccvrp')

    best = ACS(problem, iterations, DEBUG)

    # for root, directories, filenames in os.walk("instances/GoldenWasilKellyAndChao_0.1/"):
    #     for filename in filenames:
    #         file = os.path.join(root, filename)
    #         problem = loadInstance(str(file))

    #         if TIMER_MODE:
    #             run = 1
    #         else:
    #             run = 10

    #         print('name: ', problem.name, ' dimention: ',
    #               problem.dimention, ' capacity: ', problem.capacity)
    #         answers = []

    #         for _ in range(run):
    #             start = time.time()

    #             best = Colony(problem, initialPop, MAX_GENERATION,
    #                           MUTATION_RATE, DEBUG)

    #             duration = str(time.time() - start)[0:6]
    #             print('time: ', duration, 'cost: ',
    #                   round(best.cost, 2), 'sol: ', best.solution)

    #             answers.append(
    #                 (best.solution, best.cost, duration))

    #         printResult(answers)
    #         if EXEl_WRITE:
    #             writeResultToExel(filename, answers, myRow)
    #             myRow += 1
