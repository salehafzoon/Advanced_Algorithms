# import tsplib95
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


MUTATION_RATE = 0.2
POPULATION_SIZE = 50
MAX_GENERATION = 200
DEBUG = True

EXEl_WRITE = True


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
        self.depotCluster = 1
        self.clusters = []

    def __str__(self):
        return 'depot cluster: ' + str(self.depotCluster)

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

        problem.depotCluster = int(lines[cludSecIndex].split()[1])

        for i in range(problem.dimention):
            data = lines[cordSecIndex+i].split()
            node = Node(int(data[0]), int(data[1]), int(data[2]))

            node.demand = int(lines[demSecIndex+i].split()[1])
            node.cluster = int(lines[cludSecIndex+i].split()[1])
            problem.clusters.setdefault(node.cluster, []).append(node)

        # print(problem)

    return problem


class Individual(object):

    problem = None
    MList = None

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.callFitness()
        self.feasible = True

    @classmethod
    def setProblem(cls, problem):
        cls.problem = problem

    def isFeasible(self):
        routes = self.chromosome.split(str(problem.depotCluster))
        for route in routes:
            routeDemand = 0
            for c in list(route):
                nodes = problem.clusters[int(c)]
                for node in nodes:
                    routeDemand += node.demand
            if routeDemand > self.problem.capacity:
                # print('not feasible')
                return False
        return True

    @classmethod
    def createChromosome(cls):
        demands = []
        chromosome = ''

        for key in cls.problem.clusters:
            clust_demand = 0
            for node in cls.problem.clusters[key]:
                clust_demand += node.demand
            demands.append((key, clust_demand))
        # print(demands)

        depoClust = demands.pop(0)[0]
        amount = 0
        for _ in range(len(demands)):
            (cluster, demand) = rn.choice(demands)
            demands.remove((cluster, demand))

            amount += demand
            if(amount < cls.problem.capacity):
                chromosome += str(cluster)
            else:
                chromosome += str(depoClust) + str(cluster)
                amount = demand

        # print(chromosome)
        return chromosome

    def mutate(self):

        globalRoutes = self.chromosome.split(str(problem.depotCluster))
        # print(self.chromosome, globalRoutes)

        r1 = rn.randrange(len(globalRoutes))
        r2 = rn.randrange(len(globalRoutes))
        while r1 == r2:
            r2 = rn.randrange(len(globalRoutes))

        n1 = rn.randrange(len(globalRoutes[r1]))
        n2 = rn.randrange(len(globalRoutes[r2]))

        temp = globalRoutes[r1][n1]

        l1 = list(globalRoutes[r1])
        l1[n1] = globalRoutes[r2][n2]

        l2 = list(globalRoutes[r2])
        l2[n2] = temp

        globalRoutes[r1] = l1
        globalRoutes[r2] = l2

        self.chromosome = ''
        for r in globalRoutes:
            self.chromosome += ''.join(r) + str(problem.depotCluster)

        self.chromosome = self.chromosome[:-1]
        
        if(self.isFeasible()):
            self.fitness = self.callFitness()

    def crossOver(self, parent2):
        pass

    def callFitness(self):
        fitness = 0
        # print(self.chromosome)

        routes = self.chromosome.split(str(problem.depotCluster))

        depot = problem.clusters[problem.depotCluster][0]

        for route in routes:
            nodeList = [depot]

            for c in list(route):

                nodes = problem.clusters[int(c)]
                # print('cluster ', c, ':', nodes)

                for node in nodes:
                    nodeList.append(node)

            # print(depot)
            # print('all nodes:', nodeList)

            # now we have all nodes of one route
            # initial matrix of edges
            size = len(nodeList)
            mat = np.zeros(shape=(size, size))

            for i in range(size):
                for j in range(size):
                    if i != j:
                        n1 = nodeList[i]
                        n2 = nodeList[j]
                        distance = math.sqrt(
                            (n1.x - n2.x)**2 + (n1.y - n2.y)**2)

                        # nodes from the same cluster
                        if n1.cluster == n2.cluster:
                            mat[i][j] = distance

                        # adding M to distance of nodes from different clusters
                        else:
                            M = self.MList[n1.cluster][n2.cluster]
                            mat[i][j] = distance + M
                    else:
                        mat[i][j] = 0

            # print(mat)

            # now solving tsp of edges matrix
            solver = TSP()
            solver.read_mat(mat)

            sol = NN_solver()
            answer = solver.get_approx_solution(sol)

            # print('answer:', answer)
            cost = answer[0]
            nodes = answer[1]

            # decreasing total added Ms to answer
            for i in range(len(nodes)-1):
                n1 = nodeList[nodes[i]]
                n2 = nodeList[nodes[i+1]]
                
                if n1.cluster != n2.cluster:
                    cost -= self.MList[n1.cluster][n2.cluster]    

            # print('cost:', cost)
            # print('----------------------------')

            fitness += cost

        # print('fitness: ', fitness)
        return fitness

    def __str__(self):
        return self.chromosome[:10] + ' ...\t' +\
            'fitness: ' + str(self.fitness)

    def __repr__(self):
        return str(self)

    @classmethod
    def calculateMs(cls):
        clusterNum = len(cls.problem.clusters)
        mList = np.zeros(shape=(clusterNum, clusterNum))
        print('clusters:', clusterNum)

        for i in range(clusterNum):
            m = 0
            for j in range(clusterNum):
                if i != j:
                    for n1 in cls.problem.clusters[i]:
                        for n2 in cls.problem.clusters[j]:
                            m += math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

                    mList[i][j] = m
                else:
                    mList[i][j] = 0

        cls.MList = mList


def initialPop(problem):
    population = []
    Individual.setProblem(problem)
    Individual.calculateMs()

    for _ in range(POPULATION_SIZE):
        chrom = Individual.createChromosome()
        indiv = Individual(chrom)
        population.append(indiv)

    return population


def GA(problem, initialPop, maxGeneration=1000,
       mutation_rate=10, debug=True):

    population = initialPop(problem)

    for indiv in population:
        # print('before: ',population[0])
        population[0].mutate()
        # if population[0].isFeasible:
        #     print('after: ',population[0])
    

    # population[0].crossOver(population[1])


if __name__ == '__main__':


    problem = loadInstance("instances/Marc/a-n14-c4.ccvrp")

    GA(problem, initialPop, MAX_GENERATION, MUTATION_RATE, DEBUG)

    # myRow = 50
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
