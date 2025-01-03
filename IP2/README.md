Matthew Giorgio
CS534 
6/25/23
IP2

RomaniaCityApp.py
	- Gets user inputs for:
		- Start City
		- End City
		- If user wants to find another path
	- Creates the Problem and calls the 4 search algorithms

search_algorithms.py
	Contains the 4 search algorithms
	- Greedy Best-First Search
		- Chooses the neighbor with the shortest Straight Line Distance to 
			goal city
	- A* Search
		- Chooses neighbor with lowest f(n) where f(n) = h(n) + g(n)
			- g(n) = Path cost to get from start to n
			- h(n) = SLD to goal city (Same as the greedy search)
	- Hill Climbing
		- Chooses highest value neighbor based on SLD to goal
		** To me it seems like this implementation is very similar to the greedy search,
			but the textbook says the function should climb "to the smallest heuristic 
			distance to the goal"
	- Simulated Annealing
		- Chooses the neighbor to explore at random
		- There is a high level of variance in the results since it
			relies on probability
		- Doesn't always find a good path to the end

util_functions.py
	Contains miscellaneous functions (many predefined) that are 
	used throughout the code. Also defines the romania map

SimpleProblemSolvingAgent.py
	- Class Node:
		- Defines all methods that can be used on Nodes
	- Class SPSA:
		- Defines all relevent functions used by the searching algorithms
			to solve the problem
			- exp_schedule, heuristic functions etc.
