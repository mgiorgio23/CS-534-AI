import numpy as np
from util_functions import *
 
class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        path_list = [node.state for node in self.path()[:]]
        display_text = ''
        for node in path_list:
            display_text += str(f'{node} --> ')
        return display_text.strip('--> ')

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

class SPSA():

    def __init__(self, graph, initial, goal):
        self.graph = graph
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        '''Returns a list of neighboring nodes'''
        possible_actions = list(self.graph.get(state).keys())
        return possible_actions

    def result(self, state, action):
        '''In this case the action is just moving to a new node, so
        "action" is just the destination node'''
        return action

    def goal_test(self, state):
        '''Return whether or not the current state matches the goal'''
        return state == self.goal

    def path_cost(self, c, state1, action):
        '''Adds the cost to travel to the new node to the total path cost'''
        return c + self.graph.graph_dict[state1][action]
        
    def value(self, state):
        """Returns a negative number since Hill Climb and Annealing look
        to maximize this value."""
        current = self.graph.locations[state.state]
        goal = self.graph.locations[self.goal]
        distance = np.sqrt((current[0]-goal[0])**2+(current[1]-goal[1])**2)
        return -1 * distance
    
    def f(self, node: Node):
        '''Returns the straight line distance to the goal'''
        current = self.graph.locations[node.state]
        goal = self.graph.locations[self.goal]
        distance = np.sqrt((current[0]-goal[0])**2+(current[1]-goal[1])**2)
        return distance

    def exp_schedule(t, k=20, lam=0.005, limit=100):
        """Schedule function for simulated annealing"""
        return k * np.exp(-lam * t) if t < limit else 0