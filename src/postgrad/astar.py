from heapq import heappop, heappush


def reconstruct_path(came_from, current):
    """Reconstructs the path from the start to the goal."""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        if current is None:
            break

        total_path.append(current)

    return total_path[::-1]


def astar(distance, heuristic, neighbors, is_goal, start):
    """A* pathfinding algorithm.

    Args:
        distance: Function to calculate the distance between two states.
        heuristic: Function to estimate the cost from a state to the goal.
        neighbors: Function to get the neighboring states of a given state.
        is_goal: Function to check if a state is the goal.
        start: The starting state.
    """
    frontier = []
    heappush(frontier, (0, 0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, _, x = heappop(frontier)

        if is_goal(x):
            return reconstruct_path(came_from, x)

        for y in neighbors(x):
            new_cost = cost_so_far[x] + distance(x, y)
            if new_cost < cost_so_far.get(y, float("inf")):
                cost_so_far[y] = new_cost
                h = heuristic(y)
                priority = new_cost + h
                heappush(frontier, (priority, h, y))
                came_from[y] = x

    return None
