# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

from queue import PriorityQueue
import math

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function

    queue = []
    explored = set()
    queue.append(maze.start)
    while queue:
        if queue[0] == maze.start:
            cur_path = [queue.pop(0)]
        else:
            cur_path = queue.pop(0)
        cur_row, cur_col = cur_path[-1]
        if (cur_row, cur_col) in explored:
            continue
        explored.add((cur_row, cur_col))
        if maze[cur_row, cur_col] == maze.legend.waypoint:
            return cur_path
        for item in maze.neighbors_all(cur_row, cur_col):
            if item not in explored:
                queue.append(cur_path + [item])

    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    explored = set()
    pq = PriorityQueue()
    explored.add(maze.start)
    end = maze.waypoints[0]

    pq.put((0, [maze.start]))

    while not pq.empty():
        cur_cost, cur_path = pq.get()
        #print(cur_cost, cur_path);
        cur_row, cur_col = cur_path[-1]

        # If we reach the end point, return the path
        if (cur_row, cur_col) == end:
            return cur_path

        # Explore neighboring cells
        for item in maze.neighbors_all(cur_row, cur_col):
            if item not in explored:
                #print(item)
                explored.add(item)
                #print(explored)
                h = math.sqrt((item[0] -  end[0])**2 + (item[1] -  end[1])**2)
                g = len(cur_path)
                pq.put((h + g, cur_path + [item]))
                #print((h + g, cur_path))
    # If no path is found, return an empty list
    return []

def astar_single_(maze, start, end):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    if start == end:
        return []
    explored = set()
    pq = PriorityQueue()
    explored.add(start)

    pq.put((0, [start]))

    while not pq.empty():
        cur_cost, cur_path = pq.get()
        #print(cur_cost, cur_path);
        cur_row, cur_col = cur_path[-1]

        # If we reach the end point, return the path
        if (cur_row, cur_col) == end:
            return cur_path

        # Explore neighboring cells
        for item in maze.neighbors_all(cur_row, cur_col):
            if item not in explored:
                #print(item)
                explored.add(item)
                #print(explored)
                h = math.sqrt((item[0] -  end[0])**2 + (item[1] -  end[1])**2)
                g = len(cur_path)
                pq.put((h + g, cur_path + [item]))
                #print((h + g, cur_path))
    # If no path is found, return an empty list
    return []

def find_path(maze, start, end):
    explored = set()
    pq = PriorityQueue()
    pq.put((0, [start]))
    while pq:
        cur_cost, cur_path = pq.get()
        cur_row, cur_col = cur_path[-1]
        if (cur_row, cur_col) == end:
            return cur_path
        for item in maze.neighbors_all(cur_row, cur_col):
            if item not in explored:
                explored.add(item)
                h = math.sqrt((item[0] -  end[0])**2 + (item[1] -  end[1])**2)
                g = len(cur_path)
                pq.put((h + g, cur_path + [item]))

    return []

def closestdot(maze, waypoints, current):
    smallest_distance = float('inf')
    closest_dot = None
    for waypoint in waypoints:
        #h = len(astar_single_(maze, current, waypoint))
        h = math.sqrt((current[0] -  waypoint[0])**2 + (current[1] -  waypoint[1])**2)
        if h < smallest_distance and h != 0:
            smallest_distance = h
            closest_dot = waypoint
    return closest_dot

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    waypoints = maze.waypoints
    queue = list(waypoints)
    start = maze.start
    max_dis = 0
    dic = {}
#     for current in waypoints:
#         for waypoint in waypoints:
#             h = len(astar_single_(maze, current, waypoint))
#             if h < smallest_distance and h != 0:
#                 smallest_distance = h
#                 dic[current] = [h, waypoint]
    for waypoint in queue:
        h = math.sqrt((start[0] - waypoint[0])**2 + (start[1] - waypoint[1])**2)
        if h > max_dis:
            max_dis = h
            fur_end = waypoint
    queue.remove(fur_end)
    stack = []
    stack.append(fur_end)
    current = fur_end
    path = []
    while queue:
        temp = closestdot(maze, queue, current)
        stack.append(temp)
        current = temp
        queue.remove(temp)
    stack.append(maze.start)
    while len(stack) > 1:
        temp_start = stack.pop()
        temp_end = stack[-1]
        path += find_path(maze, temp_start, temp_end)
        path.pop()
    path += [fur_end]
    return path
#     start = maze.start
#     frontier = PriorityQueue()
#     visited_target = []
#     tmp = 1000
#     for next_target in maze.waypoints:
#         dist = manhattan_distance(start,next_target)
#         if dist < tmp:
#             tmp = dist
#             target = next_target

#     frontier.put((0,start))
#     came_from = {}
#     cost_so_far = {}
#     came_from[start] = None
#     cost_so_far[start] = 0

#     while frontier:
#         cur = frontier.get()[1] # Fetch the (x,y)
#         if cur == target:
#             visited_target.append(target)
#             tmp = 1000
#             if visited_target == maze.waypoints:
#                 break
#             for next_target in maze.waypoints:
#                 dist = manhattan_distance(cur,next_target)
#                 if dist < tmp and next_target not in visited_target:
#                     tmp = dist
#                     target = next_target
            

#         for i in maze.neighbors_all(cur[0], cur[1]):
#             new_cost = cost_so_far[cur] + 1  # cost to move to the next cell is always 1 in our maze

#             if i not in cost_so_far or new_cost < cost_so_far[i]:
#                 cost_so_far[i] = new_cost
#                 priority = new_cost + manhattan_distance(target, i)
#                 frontier.put((priority,i) )
#                 came_from[i] = cur

#     cur = target
#     path = [cur]

#     while cur != start:
#         cur = came_from[cur]
#         path.append(cur)

#     path.reverse()

#     return path