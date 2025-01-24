#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import time

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, rand_area,
                 expand_dis=1.0):
        """
        start: Start node [x, y, z]
        goal: Goal node [x, y, z]
        rand_area: Random sampling area [min, max]
        """

        self.MAX_ITER = 5000
        self.CONNECT_RADIUS = 2.0

        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        self.expand_dis = expand_dis
        self.node_list = [self.start]

    def planning(self):
        
        for i in range(self.MAX_ITER):

            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.is_feasible_path(nearest_node, new_node):
                near_nodes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, nearest_node, near_nodes)
                self.node_list.append(new_node)
                self.rewire(new_node, near_nodes)

            if self.reached_goal(new_node):
                return self.generate_final_course(new_node)

        return None

    # Get random node anywhere in the map, with the goal node being 
    # selected with a probability to bias the search towards the goal.
    def get_random_node(self):

        # probability for aerial node
        # random x,y
        # z is random between terrain heitght and max_rand 

        # probability for ground node
        # random x,y
        # z is terrain height

        # probability for goal node
        
        GOAL_SAMPLE_RATE = 5 # chance of selecting the goal as the random node
        if random.randint(0, 100) > GOAL_SAMPLE_RATE:
            rnd = Node(random.uniform(self.min_rand, self.max_rand),
                       random.uniform(self.min_rand, self.max_rand),
                       random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.end.x, self.end.y, self.end.z)
        return rnd


    def get_nearest_node(self, rnd_node):
        nearest_node = min(
            self.node_list, 
            key=lambda node: self.calc_distance(node, rnd_node)
        )
        return nearest_node



    # If the distance between nearest_node and rnd_node is greater than expand_dis,
    # new_node will be the point on the line connecting nearest_node and rnd_node
    # that is expand_dis away from nearest_node.
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.z)

        def calc_distance_and_angle(from_node, to_node):
            dx = to_node.x - from_node.x
            dy = to_node.y - from_node.y
            dz = to_node.z - from_node.z
            d = self.calc_distance(from_node, to_node)
            theta = math.atan2(dy, dx)
            phi = math.atan2(math.sqrt(dx**2 + dy**2), dz)
            return d, theta, phi
            
        d, theta, phi = calc_distance_and_angle(from_node, to_node)

        extend_length = min(self.expand_dis, d)

        new_node.x += extend_length * math.cos(theta) * math.sin(phi)
        new_node.y += extend_length * math.sin(theta) * math.sin(phi)
        new_node.z += extend_length * math.cos(phi)

        new_node.parent = from_node

        # Update cost to reflect the locomotion time
        # Cost to move distance d depends on the locomotion mode
        # driving/flying/swimming
        # and whether the robot has to recharge while moving
        # implement get_cost function

        new_node.cost = from_node.cost + extend_length
        
        return new_node



    def is_feasible_path(self, node1, node2):

        # if path goes below terrain height, it is not feasible
        # if path goes above terrain height, it is feasible

        return True


    def find_near_nodes(self, new_node):
        near_nodes = len(self.node_list) + 1

        # Reduce the search radius as the tree grows
        r = self.CONNECT_RADIUS * math.pow((math.log(near_nodes) / near_nodes), 1/3)
        
        near_nodes_list = [ node for node in self.node_list 
                            if self.calc_distance(node, new_node) <= r]

        return near_nodes_list


    # def choose_parent(self, neighbors, nearest_node, new_node):
    #     """Choose the best parent for the new node based on cost."""
    #     min_cost = nearest_node.cost + np.linalg.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y])
    #     best_node = nearest_node

    #     for neighbor in neighbors:
    #         cost = neighbor.cost + np.linalg.norm([new_node.x - neighbor.x, new_node.y - neighbor.y])
    #         if cost < min_cost and self.is_collision_free(neighbor):
    #             best_node = neighbor
    #             min_cost = cost

    #     new_node.cost = min_cost
    #     new_node.parent = best_node
    #     return new_node

        
    def choose_parent(self, new_node, nearest_node, near_nodes):
        min_cost = nearest_node.cost + self.calc_distance(nearest_node, new_node)
        best_node = nearest_node

        for near_node in near_nodes:
            if self.is_feasible_path(near_node, new_node):
                cost = near_node.cost + self.calc_distance(near_node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_node = near_node

        new_node.cost = min_cost
        new_node.parent = best_node

        return new_node

        # if not near_nodes:
        #     return new_node

        # costs = []
        # feasible_nodes = []
        
        # for near_node in near_nodes:
        #     t_node = self.steer(near_node, new_node, self.expand_dis)
        #     if self.is_feasible_path(near_node, t_node):
        #         cost = near_node.cost + self.calc_distance(near_node, t_node)
        #         costs.append(cost)
        #         feasible_nodes.append(near_node)
        #     else:
        #         costs.append(float('inf'))

        # if not feasible_nodes:
        #     return new_node

        # min_cost = min(costs)
        # if min_cost == float('inf'):
        #     return new_node

        # min_ind = costs.index(min_cost)
        # best_parent = feasible_nodes[min_ind]

        # new_node = self.steer(best_parent, new_node, self.expand_dis)
        # new_node.cost = min_cost
        # new_node.parent = best_parent
        # return new_node


    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.is_feasible_path(new_node, near_node):
                cost = new_node.cost + self.calc_distance(new_node, near_node)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = cost

    def calc_distance(self, node1, node2):
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        dz = node1.z - node2.z
        return math.sqrt(dx**2 + dy**2 + dz**2)


    def reached_goal(self, node):
        return self.calc_distance(node, self.end) <= self.expand_dis


    def generate_final_course(self, last_node):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = last_node
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])
        return path






def draw_graph(rrt_star, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all edges
    for node in rrt_star.node_list:
        if node.parent:
            ax.plot([node.x, node.parent.x],
                    [node.y, node.parent.y],
                    [node.z, node.parent.z],
                    color="blue", linewidth=0.5)

    # Plot path
    if path:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2], color="red", linewidth=2)

    # Plot start and goal
    ax.scatter(rrt_star.start.x, rrt_star.start.y, rrt_star.start.z, color="green", s=100, label="Start")
    ax.scatter(rrt_star.end.x, rrt_star.end.y, rrt_star.end.z, color="magenta", s=100, label="Goal")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def main():
    print("Start RRT* path planning in 3D")

    # Define start and goal positions
    start = [0, 0, 0]
    goal = [1000, 1000, 1000]

    # Define random sampling area
    rand_area = [-50, 1050]

    # Initialize RRT*
    rrt_star = RRTStar(start, goal, rand_area,
                       expand_dis=20.0)

    # Run planning
    start_time = time.perf_counter()    
    path = rrt_star.planning()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Planning method runtime: {elapsed_time:.6f} seconds")

    # Draw graph
    if path:
        print("Path found!")
        draw_graph(rrt_star, path)
    else:
        print("No path found")
        draw_graph(rrt_star)

if __name__ == '__main__':
    main()


#%%