import tsplib95
from pprint import pprint
import random as rn
import math
import copy
from random import randrange
import matplotlib.pyplot as plt
import time
import numpy as np
import openpyxl
import os

MUTATION_RATE = 0.2
POPULATION_SIZE = 70
MAX_GENERATION = 200
DEBUG = False

EXEl_WRITE = True


class Node(object):

    def __init__(self, number, x, y):
        self.number = number
        self.x = x
        self.y = y
        self.demand = None
        self.cluster = None

    def __str__(self):
        return str(self.number)

    def __repr__(self):
        return str(self)


class Problem(object):

    def __init__(self):
        self.name = None
        self.dimention = None
        self.capacity = None
        self.depot = 1
        self.clusters = []

    def __str__(self):
        return str('dimention:', self.dimention)

    def __repr__(self):
        return str(self)


def printResult(answers):

    minAns = min(answers, key=lambda t: t[1])
    maxAns = max(answers, key=lambda t: t[1])
    variance = round(math.sqrt(np.var([ans[1]for ans in answers])), 3)

    print("\nbest[0:10]=", minAns[0][0:10], "\tmin cost:", minAns[1])
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
    maxCost = max(answers, key=lambda t: t[1])[1]
    avgCost = sum(ans[1] for ans in answers)/len(answers)
    costVariance = round(math.sqrt(np.var([ans[1]for ans in answers])), 3)

    minTime = min(answers, key=lambda t: t[2])[2]
    maxTime = max(answers, key=lambda t: t[2])[2]
    avgTime = str(sum(float(ans[2])for ans in answers)/len(answers))[0:6]

    wbkName = 'Results.xlsx'
    wbk = openpyxl.load_workbook(wbkName)
    for wks in wbk.worksheets:
        myCol = 4

        wks.cell(row=myRow, column=1).value = file_name

        wks.cell(row=myRow, column=myCol).value = minCost
        wks.cell(row=myRow, column=myCol+1).value = avgCost
        wks.cell(row=myRow, column=myCol+2).value = maxCost
        wks.cell(row=myRow, column=myCol+3).value = costVariance

        wks.cell(row=myRow, column=myCol+4).value = minTime
        wks.cell(row=myRow, column=myCol+5).value = avgTime
        wks.cell(row=myRow, column=myCol+6).value = maxTime

    wbk.save(wbkName)
    wbk.close


def plotResult(costs):

    plt.plot(list(range(len(costs))), costs, '-', color="blue",
             label='algorithm progress', linewidth=2)
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

        for i in range(problem.dimention):
            data = lines[cordSecIndex+i].split()
            node = Node(int(data[0]), int(data[1]), int(data[2]))

            node.demand = int(lines[demSecIndex+i].split()[1])
            node.cluster = int(lines[cludSecIndex+i].split()[1])
            problem.clusters.setdefault(node.cluster, []).append(node)

        # print(problem.clusters)
    return problem

def initialPop(problem):
    demands = []
    for key in problem.clusters:
        clust_demand = 0
        for node in problem.clusters[key]:
            clust_demand += node.demand
        demands.append((key , clust_demand))
    # print(demands)

    

def fitnessFunc(problem):
    pass

def crossOver(problem):
    pass

def mutate(problem):
    pass

def GA(problem, initialPop, fitnessFunc, crossOver, mutate, maxGeneration=1000,
       mutation_rate=10, debug=True):
    
    population = initialPop(problem)

if __name__ == '__main__':

    problem = loadInstance("instances/Marc/a-n14-c4.ccvrp")
    initialPop(problem)

    myRow = 50
    # for root, directories, filenames in os.walk("instances/M"):
    #     for filename in filenames:
    #         file = os.path.join(root, filename)
    #         problem = tsplib95.load_problem(str(file))

    #         for _ in range(10):
    #             # start = time.time()

    #             pass

    #             # duration = str(time.time() - start)[0:6]
    #             # answers.append((state, cost, duration))

    #         # printResult(answers)
    #         # if EXEl_WRITE:
    #         #     writeResultToExel(filename, answers, myRow)
    #         #     myRow += 1
