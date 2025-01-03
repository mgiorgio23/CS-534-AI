from util_functions import romania_map
from SimpleProblemSolvingAgent import Node, SPSA
from search_algorithms import hill_climbing, best_first_graph_search, astar_search, simulated_annealing

def get_start_city():
    '''Loops until user enters valid start city.
    Using .title() to cover a larger range of inputs'''
    start_city = input('Please enter the origin city: ').title()
    while True:
        if start_city not in romania_map.nodes():
            start_city = input(f'Could not find {start_city}, please try again: ').title()
        else: 
            return start_city

def get_end_city(start_city):
    '''Loops until user enters valid end city'''
    end_city = input('Please enter the destination city: ').title()
    while True:
        if end_city not in romania_map.nodes():
            end_city = input(f'Could not find {end_city} please try again: ').title()
        elif end_city == start_city:
            end_city = input(f'Destination cannot be the same as the start. Please try again: ').title()
        else:
            print('\n')
            return end_city
    
def get_rerun_answer():
    '''Loops until user gives a valid response'''
    run = input('Would you like to find the best path between another two cities? ')
    while True:
        if run.lower() == 'yes':
            break
        elif run.lower() == 'no':
            run = False
            break
        else:
            run = input('Please enter "yes" or "no"? ')
    return run

def main():
    '''Main loop that runs until user terminates'''
    while True:
        #Get inputs
        start_city = get_start_city()
        end_city = get_end_city(start_city)

        ##Greedy First Search
        # f is the heuristic function defined in SPSA
        greedy = SPSA(romania_map, start_city, end_city)
        print('\nGreedy Best-First Search')
        best_first_graph_search(greedy, greedy.f)

        ## A* Search
        ## In this case 'h' is the same function 'f' that the greedy search uses
        astar = SPSA(romania_map, start_city, end_city)
        print('A* Search')
        astar_search(astar, astar.f)

        ## Hill Climbing Search
        print('Hill Climbing Search')
        climb = SPSA(romania_map, start_city, end_city)
        hill_climbing(climb)

        ## Simulated Annealing Search
        print('Simulated Annealing Search')
        annealing = SPSA(romania_map, start_city, end_city)
        simulated_annealing(annealing)
        
        #Check if the user wants to find another path, if not, end loop
        if not get_rerun_answer():
            break
        print('\n')

    print('\nThank You for Using Our App')

if __name__ == "__main__":
    main()

