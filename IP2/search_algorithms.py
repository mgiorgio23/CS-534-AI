import sys
from util_functions import *
from SimpleProblemSolvingAgent import *

def best_first_graph_search(problem, f, display=False):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            print(node.solution())
            print(f'Total Cost: {node.path_cost}\n')
            #print(problem.cost)
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)

    return None


def astar_search(problem, h=None, display=False):
    """h(n) is the same function uses in Greedy Search
    g(n) is the path cost to get from the start node to node n"""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


def hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node))
        '''I have it calculating a positive distance so I only want to stop 
        when I can only get further away from goal'''
        if problem.value(neighbor) <= problem.value(current):
            break
        current = neighbor
    print(current.solution())
    print(f'Total Cost: {current.path_cost}\n')
    return current.state

def exp_schedule(t, k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return k * np.exp(-lam * t) if t < limit else 0

def simulated_annealing(problem):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = exp_schedule(t)
        if T == 0:
            print(current.solution())
            print(f'Total Cost: {current.path_cost}\n')
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice) - problem.value(current)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice