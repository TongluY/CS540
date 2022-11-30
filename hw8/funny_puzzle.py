import heapq




def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(9):
        if to_state[i] == 0: 
            continue
        target = from_state.index(to_state[i])
        x0, x1 = target // 3, i // 3
        y0, y1 = target % 3, i % 3
        distance += abs(x0 - x1) + abs(y0 - y1)
    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    succ_states = []
    def val(x,y):
        return x>=0 and x<=2 and y>=0 and y<=2
    for i in range(9):
        if state[i] == 0:
            x0, y0 = i // 3, i % 3
            for x, y in directions:
                x1, y1 = x0 + x, y0 + y
                ni = x1 * 3 + y1
                if val(x1, y1) and state[ni]:
                    new_l = state[:]
                    new_l[i] = new_l[ni]
                    new_l[ni] = 0
                    succ_states.append(new_l)
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    # cost, state, (g, h, parent_index)
    move, ind, length = 0, 0, 1
    h = get_manhattan_distance(state, goal_state)
    open, closed, cost, result = [], [], {}, [[state, h]]
    heapq.heappush(open, (0 + h, state, (0, h, -1)))
    while open:
        curr = heapq.heappop(open)
        curr_s = curr[1]
        cost[ind] = curr
        closed.append(curr_s)
        if curr_s == goal_state:
            x = curr
            break
        length -= 1
        for i in get_succ(curr_s):
            if i not in closed:
                hi = get_manhattan_distance(i)
                gi = curr[2][0] + 1
                fi = hi + gi
                length += 1
                heapq.heappush(open, (fi, i, (gi, hi, ind)))
        ind += 1
    def trace(x):
        if x[2][2] > -1:
            trace(cost[x[2][2]])
            result.append([x[1], x[2][1]])
    trace(x)
    for i in result:
        print(i[0], 'h={}'.format(i[1]), "moves:", move)
        move += 1
    print("Max queue length:", length)


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()
    solve([4,3,0,5,1,6,7,2,0])

    solve([2,5,1,4,0,6,7,0,3])
    print()
