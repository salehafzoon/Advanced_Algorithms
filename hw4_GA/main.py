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


LINEAR = 'linear'
LOG = 'logarithmic'
EXP = 'exponential'

START_T = 1
T = START_T
ALPHA = 0.9
TEMP_MODE = EXP
INIT_HEURISTIC = True
NUM_ITERATIONS = 500
DEBUG = False
EPSILON = 1e-323
graph = None
dependencies = []

EXEl_WRITE = True


def printResult(answers):

    minAns = min(answers, key=lambda t: t[1])
    maxAns = max(answers, key=lambda t: t[1])
    variance = round(math.sqrt(np.var([ans[1]for ans in answers])) , 3)

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
    costVariance = round(math.sqrt(np.var([ans[1]for ans in answers])) , 3)

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


if __name__ == '__main__':

    problem = tsplib95.load_problem("instances/Marc/a-n14-c4.ccvrp")

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
