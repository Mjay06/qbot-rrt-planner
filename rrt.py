"""RRT implementation from Lab 2, with a planner wrapper for the QBot project."""
import math
from typing import List, Tuple

import numpy as np


def conf_free(q: np.ndarray,
              obstacles: List[Tuple[np.ndarray, float]]) -> bool:
    """Return whether a configuration lies outside all circular obstacles."""
    x, y = q[0], q[1]

    for center, radius in obstacles:
        a, b = center[0], center[1]
        distance_squared = (x - a) ** 2 + (y - b) ** 2
        if distance_squared <= radius ** 2:
            return False

    return True


def edge_free(edge: Tuple[np.ndarray, np.ndarray],
              obstacles: List[Tuple[np.ndarray, float]]) -> bool:
    """Return whether a line segment lies entirely outside all obstacles."""
    A, B = edge
    AB = B - A

    for center, radius in obstacles:
        a = np.dot(AB, AB)
        b = 2 * np.dot(A - center, AB)
        c = np.dot(A - center, A - center) - radius ** 2

        if a == 0:
            if np.linalg.norm(A - center) <= radius:
                return False
            continue

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            continue

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        if (t1 > 1 and t2 > 1) or (t1 < 0 and t2 < 0):
            continue
        return False

    return True


def random_conf(width: float, height: float) -> np.ndarray:
    """Sample a configuration uniformly from the rectangular workspace."""
    x_rand = np.random.rand() * width
    y_rand = np.random.rand() * height
    return np.array([x_rand, y_rand])


def nearest_vertex(conf: np.ndarray, vertices: np.ndarray) -> int:
    """Return the index of the vertex nearest to ``conf``."""
    x, y = conf
    distance_shortest = None
    index = None

    for i, vertex in enumerate(vertices):
        a, b = vertex
        distance = math.sqrt((x - a) ** 2 + (y - b) ** 2)

        if distance_shortest is None:
            distance_shortest = distance
            index = i
        elif distance_shortest > distance:
            distance_shortest = distance
            index = i

    return index


def extend(origin: np.ndarray, target: np.ndarray,
           step_size: float = 0.2) -> np.ndarray:
    """Extend from ``origin`` toward ``target`` by at most ``step_size``."""
    direction = target - origin
    distance = np.linalg.norm(direction)
    if distance < step_size:
        return target.copy()
    t = step_size / distance
    return origin + t * direction


def rrt(origin: np.ndarray, width: float, height: float,
        obstacles: List[Tuple[np.ndarray, float]], trials: int = 1500,
        step_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Build an RRT using the Lab 2 sampling and collision-checking logic."""
    num_verts = 1

    vertices = np.zeros((trials + 1, len(origin)))
    vertices[0, :] = origin

    parents = np.zeros(trials + 1, dtype=int)
    parents[0] = -1

    for _ in range(trials):
        new_conf = random_conf(width, height)
        existing_vertices = vertices[:num_verts, :]
        parent_index = nearest_vertex(new_conf, existing_vertices)
        parent_vertex = existing_vertices[parent_index]
        vertex_new = extend(parent_vertex, new_conf, step_size)
        edge = (parent_vertex, vertex_new)
        if conf_free(vertex_new, obstacles) and edge_free(edge, obstacles):
            vertices[num_verts, :] = vertex_new
            parents[num_verts] = parent_index
            num_verts += 1

    return vertices[:num_verts, :], parents[:num_verts]


def backtrack(index: int, parents: np.ndarray) -> List[int]:
    """Return the node indices from the tree origin to ``index``."""
    path = []
    current_index = index

    while current_index != -1:
        path.append(current_index)
        current_index = parents[current_index]

    return path[::-1]


class RRT:
    """Adapt the Lab 2 RRT function to the QBot project's planner interface."""

    def __init__(self, start, goal, bounds, obstacles, obstacle_radius,
                 robot_radius, step_size=0.2, max_iter=3000, seed=None):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = np.asarray(obstacles, dtype=float).reshape((-1, 2))
        self.inflated_radius = obstacle_radius + robot_radius

        x_min, x_max, y_min, y_max = bounds
        self.x_min, self.x_max = x_min + robot_radius, x_max - robot_radius
        self.y_min, self.y_max = y_min + robot_radius, y_max - robot_radius
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.offset = np.array([self.x_min, self.y_min])
        self.step_size = step_size
        self.max_iter = max_iter
        self.seed = seed
        self.nodes = [self.start]
        self.parents = [-1]
        self._local_obstacles = [
            (center - self.offset, self.inflated_radius)
            for center in self.obstacles
        ]

    def _to_local(self, point):
        return np.asarray(point, dtype=float) - self.offset

    def point_free(self, point):
        if not (self.x_min <= point[0] <= self.x_max and
                self.y_min <= point[1] <= self.y_max):
            return False
        return conf_free(self._to_local(point), self._local_obstacles)

    def edge_free(self, a, b):
        return (self.point_free(a) and self.point_free(b) and
                edge_free((self._to_local(a), self._to_local(b)),
                          self._local_obstacles))

    def plan(self):
        """Return start-to-goal waypoints, or ``None`` if no path is found."""
        self.nodes = [self.start]
        self.parents = [-1]

        if not self.point_free(self.start) or not self.point_free(self.goal):
            return None

        initial_state = np.random.get_state() if self.seed is not None else None
        if self.seed is not None:
            np.random.seed(self.seed)
        try:
            vertices, parents = rrt(
                self._to_local(self.start), self.width, self.height,
                self._local_obstacles, trials=self.max_iter,
                step_size=self.step_size,
            )
        finally:
            if initial_state is not None:
                np.random.set_state(initial_state)

        self.nodes = [vertex + self.offset for vertex in vertices]
        self.parents = parents.tolist()

        goal_local = self._to_local(self.goal)
        goal_index = nearest_vertex(goal_local, vertices)
        nearest = self.nodes[goal_index]
        if (np.linalg.norm(nearest - self.goal) > self.step_size or
                not self.edge_free(nearest, self.goal)):
            return None

        path = [tuple(self.nodes[i]) for i in backtrack(goal_index, parents)]
        if not np.array_equal(nearest, self.goal):
            path.append(tuple(self.goal))
            self.nodes.append(self.goal.copy())
            self.parents.append(goal_index)
        return path

    @staticmethod
    def path_length(path):
        return sum(math.dist(path[i], path[i + 1])
                   for i in range(len(path) - 1))
