import tsplib95
from pprint import pprint
import random as rn
import math
import copy
from random import randrange

LINEAR = 'linear'
LOG = 'logarithmic'
EXP = 'exponential'

START_T = 1
T = START_T
ALPHA = 0.8
TEMP_MODE = EXP
INIT_HEURISTIC = True
NUM_ITERATIONS = 500
DEBUG = False
EPSILON = 1e-323
problem = tsplib95.load_problem("instances/H/typeset.10835.26.sop")
graph = None
dependencies = []


class Edge(object):

    def __init__(self, vertices, weight):
        self.vertices = vertices
        self.weight = weight

    def __str__(self):
        return str(self.vertices) + "->" + str(self.weight)
        # return str(self.weight)
        # return str(self.vertices)

    def __repr__(self):
        return str(self)


class Graph(object):

    def __init__(self, problem):
        self.edges = []
        self.dependencies = []
        self.dimension = problem.dimension
        problemEgdes = list(problem.get_edges())
        problemWeights = problem.edge_weights[1:]

        for i in range(len(problemEgdes)):
            # if (i % 19 != 0):
            self.edges.append(Edge(problemEgdes[i], problemWeights[i]))


def calculateDependencies(problem):
    dependencies = []
    edgeWeights = problem.edge_weights[1:]

    for i in range(problem.dimension):
        dependencies.append(list())
        for j in range(graph.dimension):
            if(edgeWeights[(i*problem.dimension)+j] == -1):
                dependencies[i].append(j)
    return dependencies


def fpp3exchange(problem, deps, solution):
    dimension = problem.dimension
    edgeWeights = problem.edge_weights[1:]

    solutions = []
    for it in range(int(dimension/2)):
        h = randrange(0, dimension-3)
        i = h + 1
        leftPath = []
        leftPathLen = randrange(1, int(dimension-i))
        leftPath.extend(solution[i:i+leftPathLen])

        i += leftPathLen
        # print("left:", leftPath)
        end = False
        rightPath = []
        for j in range(i, len(solution)):

            for dep in deps[solution[j]]:
                if dep != 0 and dep in leftPath:
                    end = True
                    break

            # terminate the progress
            if end:
                break
            # add j to right path
            else:
                rightPath.append(solution[j])

        if(len(rightPath) != 0):
            # print("right:", rightPath, "\n")
            sol = solution[0:h+1]
            sol.extend(rightPath)
            sol.extend(leftPath)
            sol.extend(solution[len(sol):])
            # print("sol:", sol, "\n")
            solutions.append((sol, cost_function(problem, sol)))

    solutions.sort(key=lambda x: x[1])
    # print('\nforward:', solutions)
    if len(solutions) != 0:
        return solutions[0]
    else:
        return None


def bpp3exchange(problem, deps, solution):
    dimension = problem.dimension
    edgeWeights = problem.edge_weights[1:]

    solutions = []
    for it in range(int(dimension/2)):
        h = randrange(3, dimension)
        # h = 3
        i = h - 1
        rightPath = []
        rightPathLen = randrange(1, i+1)
        rightPath.extend(solution[i-rightPathLen+1:i+1])
        rightDeps = []

        for node in rightPath:
            rightDeps.extend(deps[node])

        i -= rightPathLen
        # print("right:", rightPath)
        # print('right deps:',rightDeps)

        leftPath = []
        for j in range(i, 0, -1):

            # add j to left path
            if solution[j] not in rightDeps:
                leftPath.insert(0, solution[j])
            else:
                break

        if(len(leftPath) != 0):
            # print("left:", leftPath)
            sol = solution[h:]
            sol = leftPath + sol
            sol = rightPath + sol
            sol = solution[:dimension - len(sol)] + sol
            # print("sol:", sol, "\n")
            solutions.append((sol, cost_function(problem, sol)))

    solutions.sort(key=lambda x: x[1])
    # print('\nbackward:', solutions)
    if len(solutions) != 0:
        return solutions[0]
    else:
        return None


def random_start(graph, deps):
    solution = []
    dependencies = copy.deepcopy(deps)

    while(len(solution) < graph.dimension):
        for i in range(graph.dimension):
            if(INIT_HEURISTIC):
                src = 0
                if len(solution) != 0:
                    src = solution[-1]

                if len(solution) == 7:
                    pass

                candidates = []

                result = [i for i in range(
                    len(dependencies)) if len(dependencies[i]) == 0]

                # print(result)

                candidates = [
                    (i, graph.edges[(src*graph.dimension) + i].weight)
                    for i in result if i not in solution]

                candidates = sorted(candidates, key=lambda tup: tup[1])
                # print(candidates)

                solution.append(candidates[0][0])

                for dep in dependencies:
                    if(candidates[0][0] in dep):
                        dep.remove(candidates[0][0])
                # print(solution)

            else:
                if(len(dependencies[i]) == 0 and not(i in solution)):
                    solution.append(i)
                    for dep in dependencies:
                        if(i in dep):
                            dep.remove(i)

    return solution


def cost_function(problem, solution):

    weight = 0
    edgeWeights = problem.edge_weights[1:]
    sol = copy.deepcopy(solution)

    while(len(sol) > 1):
        src = sol.pop(0)
        dest = sol[0]
        w = edgeWeights[(src*problem.dimension)+dest]
        # print(src, dest, w)
        weight += w

    return weight


def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:

        p = math.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p


def get_neighbour(problem, dependencies, state):
    new_states = []
    new_state1 = fpp3exchange(problem, dependencies, state)
    new_state2 = bpp3exchange(problem, dependencies, state)

    if new_state1 != None:
        new_states.append(new_state1)
    if new_state2 != None:
        new_states.append(new_state2)

    if len(new_states) != 0:
        new_states.sort(key=lambda x: x[1])
        return new_states[0]

    else:
        return state

# T_(k+1)= α T_k [Dosso,Oldenburg, 1991]
# T_(k+1)= T_0/log⁡(k+1) [Geman and Geman]
# T_k= e^(-αk).T_0
# T_k=(T_1/T_0 ) ^k ,(T_1/T_(0 ) )=0.9 [Kirptrick, 1983]


def updateTemperature(step):
    global T
    if TEMP_MODE == LINEAR:
        return ALPHA * T
    elif TEMP_MODE == LOG:
        return START_T / math.log(step+2)
    elif TEMP_MODE == EXP:
        return math.exp(-ALPHA * step+1)*START_T


def annealing(random_start, cost_function, random_neighbour,
              acceptance, updateTemperature, maxsteps=1000, debug=True):

    global problem
    global T
    state = random_start(graph, dependencies)
    cost = cost_function(problem, state)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        # print('len:', len(get_neighbour(problem, dependencies, state)))
        (new_state, new_cost) = get_neighbour(problem, dependencies, state)
        if debug:
            # print('step:', step, '\t T:', T, '\t state:',
            #       state, '\t cost:', cost, '\t new_state:', new_state,
            #       '\t new_cost:', new_cost)

            print('step:', step, '\t T:', T, '\t new_cost:', new_cost)

        if acceptance_probability(cost, new_cost, T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
        T = updateTemperature(step)
        if T == 0.0:
            T = EPSILON

    return new_state, new_cost, states, costs


if __name__ == '__main__':

    graph = Graph(problem)
    dependencies = calculateDependencies(problem)

    # print('\n', dependencies)
    # solution = random_start(graph, dependencies)
    # print('\n', solution)
    # weight = cost_function(problem, solution)
    # print("weight:", weight, "\n")

    # solutions = []
    # solutions.append(fpp3exchange(problem, dependencies, solution))
    # solutions.append(bpp3exchange(problem, dependencies, solution))

    # print("\nsolutions:", solutions)

    # best = 0

    answers = []
    for _ in range(10):
        state, cost, states, costs = annealing(random_start, cost_function, get_neighbour,
                                               acceptance_probability, updateTemperature, NUM_ITERATIONS, DEBUG)
        print(cost, 'finded')
        answers.append((state, cost))

    print("min cost:", min(answers, key=lambda t: t[1])[1])
    print("avg cost:", sum(ans[1] for ans in answers)/len(answers))
