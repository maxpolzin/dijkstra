#%% 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import time

def calculate_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

def calculate_distance_and_angle(node1, node2):
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    dz = node2.z - node1.z

    d = calculate_distance(node1, node2)

    theta = math.atan2(dy, dx)
    phi = math.atan2(math.sqrt(dx**2 + dy**2), dz)

    return d, theta, phi

def calculate_cost(from_node, to_node):
    # Placeholder: in the future incorporate terrain slope, battery usage, etc.
    return calculate_distance(from_node, to_node)

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0


class RRTStar:
    def __init__(self, start, goal, rand_area, expand_dis=1.0):

        self.MAX_ITER = 5000
        self.CONNECT_RADIUS = 2.0

        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])

        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis

        self.node_list = [self.start]

    def planning(self):

        for _ in range(self.MAX_ITER):

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


    def get_random_node(self):
        # probability for aerial node
        # random x, y
        # z is random between terrain height and max_rand 

        # probability for ground node
        # random x, y
        # z is terrain height

        # probability for goal node
        GOAL_SAMPLE_RATE = 5
        if random.randint(0, 100) > GOAL_SAMPLE_RATE:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            z = random.uniform(self.min_rand, self.max_rand)
            return Node(x, y, z)
        else:
            return Node(self.end.x, self.end.y, self.end.z)


    def get_nearest_node(self, rnd_node):
        return min(self.node_list, key=lambda node: calculate_distance(node, rnd_node))


    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.z)

        d, theta, phi = calculate_distance_and_angle(from_node, to_node)
        dist = min(extend_length, d)

        new_node.x += dist * math.cos(theta) * math.sin(phi)
        new_node.y += dist * math.sin(theta) * math.sin(phi)
        new_node.z += dist * math.cos(phi)

        new_node.parent = from_node
        new_node.cost = from_node.cost + calculate_cost(from_node, new_node)
        return new_node


    def is_feasible_path(self, node1, node2):
        # In the future integrate terrain checks or collision checks
        return True


    def find_near_nodes(self, new_node):
        n_nodes = len(self.node_list) + 1
        r = self.CONNECT_RADIUS * (math.log(n_nodes) / n_nodes)**(1.0 / 3.0)
        return [node for node in self.node_list if calculate_distance(node, new_node) <= r]


    def choose_parent(self, new_node, nearest_node, near_nodes):
        min_cost = nearest_node.cost + calculate_cost(nearest_node, new_node)
        best_node = nearest_node

        for near_node in near_nodes:
            if self.is_feasible_path(near_node, new_node):
                cost = near_node.cost + calculate_cost(near_node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_node = near_node

        new_node.cost = min_cost
        new_node.parent = best_node

        return new_node


    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.is_feasible_path(new_node, near_node):
                new_cost = new_node.cost + calculate_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def reached_goal(self, node):
        return calculate_distance(node, self.end) <= self.expand_dis

    def generate_final_course(self, last_node):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = last_node
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.reverse()
        return path

def draw_graph(rrt_star, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for node in rrt_star.node_list:
        if node.parent:
            ax.plot([node.x, node.parent.x],
                    [node.y, node.parent.y],
                    [node.z, node.parent.z],
                    color="blue", linewidth=0.5)
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="red", linewidth=2)
    ax.scatter(rrt_star.start.x, rrt_star.start.y, rrt_star.start.z,
               color="green", s=100, label="Start")
    ax.scatter(rrt_star.end.x, rrt_star.end.y, rrt_star.end.z,
               color="magenta", s=100, label="Goal")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def main():
    print("Start RRT* path planning in 3D")

    start = [0, 0, 0]
    goal = [1000, 1000, 1000]

    rand_area = [-50, 1050]

    rrt_star = RRTStar(start, goal, rand_area, expand_dis=20.0)

    start_time = time.perf_counter()
    path = rrt_star.planning()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Planning method runtime: {elapsed_time:.6f} seconds")
    
    if path:
        print("Path found!")
        draw_graph(rrt_star, path)
    else:
        print("No path found")
        draw_graph(rrt_star)

if __name__ == '__main__':
    main()
