"""Plan an RRT route and drive the virtual QBot in Quanser Interactive Labs.

Copy this file beside qlabs_setup.py in the Quanser project folder, then run:
    python qlabs_rrt_run.py
"""
import math
import time
from typing import List, Tuple

import numpy as np
from qlabs_setup import setup
from pal.products.qbot_platform import QBotPlatformDriver
from qvl.basic_shape import QLabsBasicShape
from qvl.qlabs import QuanserInteractiveLabs


# --- ENVIRONMENT MAPPING DATA ---
cage_bounds = [-2.0, 2.0, -2.0, 2.0]
pvc_centers = [
    (0.0, -1.0),
    (0.5, 0.4),
    (-0.7, 0.5),
    (0.0, 1.5),
    (0.0, 1.0),
]
r_pvc = 0.05
r_qbot = 0.17
pvc_height = 0.60
start = np.array([0.0, 0.0], dtype=float)
goal = np.array([1.5, 1.5], dtype=float)
rrt_trials = 5000
rrt_step_size = 0.2
random_seed = 42

Obstacle = Tuple[np.ndarray, float]


def conf_free(q: np.ndarray, obstacles: List[Obstacle]) -> bool:
    x, y = q[0], q[1]

    for center, radius in obstacles:
        a, b = center[0], center[1]
        distance_squared = (x - a) ** 2 + (y - b) ** 2
        if distance_squared <= radius ** 2:
            return False

    return True


def edge_free(edge: Tuple[np.ndarray, np.ndarray],
              obstacles: List[Obstacle]) -> bool:
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
    x_rand = np.random.rand() * width
    y_rand = np.random.rand() * height
    return np.array([x_rand, y_rand])


def nearest_vertex(conf: np.ndarray, vertices: np.ndarray) -> int:
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
    direction = target - origin
    distance = np.linalg.norm(direction)
    if distance < step_size:
        return target.copy()
    t = step_size / distance
    return origin + t * direction


def rrt(origin: np.ndarray, width: float, height: float,
        obstacles: List[Obstacle], trials: int = 1500,
        step_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
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
    path = []
    current_index = index

    while current_index != -1:
        path.append(current_index)
        current_index = parents[current_index]

    return path[::-1]


def plan_path() -> List[Tuple[float, float]]:
    np.random.seed(random_seed)
    x_min, x_max, y_min, y_max = cage_bounds
    x_min += r_qbot
    x_max -= r_qbot
    y_min += r_qbot
    y_max -= r_qbot
    offset = np.array([x_min, y_min])
    width = x_max - x_min
    height = y_max - y_min
    inflated_obstacles = [
        (np.asarray(center, dtype=float) - offset, r_pvc + r_qbot)
        for center in pvc_centers
    ]
    local_start = start - offset
    local_goal = goal - offset

    if not conf_free(local_start, inflated_obstacles):
        raise ValueError("The start point overlaps an inflated obstacle.")
    if not conf_free(local_goal, inflated_obstacles):
        raise ValueError("The goal point overlaps an inflated obstacle.")

    vertices, parents = rrt(
        local_start,
        width,
        height,
        inflated_obstacles,
        trials=rrt_trials,
        step_size=rrt_step_size,
    )
    goal_index = nearest_vertex(local_goal, vertices)
    nearest = vertices[goal_index]
    if (np.linalg.norm(nearest - local_goal) > rrt_step_size or
            not edge_free((nearest, local_goal), inflated_obstacles)):
        raise RuntimeError(
            "RRT did not find a collision-free connection to the goal. "
            "Increase rrt_trials or check the map."
        )

    path = [tuple(vertices[i] + offset) for i in backtrack(goal_index, parents)]
    if not np.array_equal(nearest, local_goal):
        path.append(tuple(goal))
    return path


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class QLabsQBot:
    """Drive the QBot and estimate its pose from commands and gyro readings."""

    def __init__(self, driver: QBotPlatformDriver):
        self.driver = driver
        self.x, self.y = map(float, start)
        self.heading = 0.0
        self.started_at = time.time()
        self.last_update = None

    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.heading

    def drive(self, forward_speed: float, turn_rate: float,
              duration: float) -> None:
        end_time = time.monotonic() + duration
        commands = np.array([forward_speed, turn_rate], dtype=np.float64)

        while time.monotonic() < end_time:
            new_data = self.driver.read_write_std(
                timestamp=time.time() - self.started_at,
                arm=1,
                commands=commands,
            )
            now = time.time()
            if new_data:
                if self.last_update is not None:
                    elapsed = now - self.last_update
                    self.heading = wrap_angle(
                        self.heading + self.driver.gyroscope[2] * elapsed
                    )
                    self.x += forward_speed * math.cos(self.heading) * elapsed
                    self.y += forward_speed * math.sin(self.heading) * elapsed
                self.last_update = now
            time.sleep(0.002)

    def stop(self) -> None:
        self.drive(0.0, 0.0, 0.1)


def wait_for_driver(driver: QBotPlatformDriver,
                    started_at: float,
                    timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    zero_commands = np.zeros(2, dtype=np.float64)

    while time.monotonic() < deadline:
        if driver.read_write_std(
            timestamp=time.time() - started_at,
            arm=0,
            commands=zero_commands,
        ):
            return
        time.sleep(0.05)

    raise ConnectionError(
        "No data received from the QBot driver at localhost. "
        "Check that QLabs is open and the virtual QBot driver is active."
    )


def follow_waypoints(robot: QLabsQBot,
                     path: List[Tuple[float, float]]) -> None:
    max_forward_speed = 0.3
    max_turn_rate = 1.5
    max_steps_per_waypoint = 1200

    for index, waypoint in enumerate(path[1:], start=1):
        target = np.asarray(waypoint)
        tolerance = 0.05 if index == len(path) - 1 else 0.04

        for _ in range(max_steps_per_waypoint):
            x, y, heading = robot.pose()
            dx, dy = target - np.array([x, y])
            distance = float(np.hypot(dx, dy))
            if distance < tolerance:
                break

            heading_error = wrap_angle(math.atan2(dy, dx) - heading)
            forward_speed = (
                0.0 if abs(heading_error) > 0.6
                else min(max_forward_speed, distance) * math.cos(heading_error)
            )
            turn_rate = float(np.clip(
                2.5 * heading_error, -max_turn_rate, max_turn_rate
            ))
            robot.drive(forward_speed, turn_rate, 0.05)
        else:
            raise TimeoutError(
                f"QBot did not reach waypoint {index}: {np.round(target, 2)}"
            )


def run_in_qlabs(path: List[Tuple[float, float]]) -> None:
    qlabs = None
    driver = None
    robot = None

    try:
        setup(
            locationQBotP=[float(start[0]), float(start[1]), 0.05],
            rotationQBotP=[0, 0, 0],
            verbose=True,
        )
        time.sleep(2.0)

        qlabs = QuanserInteractiveLabs()
        if not qlabs.open("localhost"):
            raise ConnectionError("Could not connect to QLabs at localhost.")

        shape_spawner = QLabsBasicShape(qlabs)
        pvc_scale = [r_pvc * 2, r_pvc * 2, pvc_height]
        print("Spawning virtual obstacles...")
        for x, y in pvc_centers:
            shape_spawner.spawn(
                location=[x, y, pvc_height / 2],
                rotation=[0, 0, 0],
                scale=pvc_scale,
                configuration=shape_spawner.SHAPE_CYLINDER,
                waitForConfirmation=True,
            )

        print("Connecting to virtual QBot...")
        driver = QBotPlatformDriver(mode=1, ip="localhost")
        wait_for_driver(driver, time.time())
        robot = QLabsQBot(driver)

        print("Following the RRT path in QLabs. Press Ctrl+C to stop.")
        follow_waypoints(robot, path)
        print("QBot reached the goal.")
    except KeyboardInterrupt:
        print("Run interrupted; stopping the QBot.")
    finally:
        try:
            if robot is not None:
                robot.stop()
        finally:
            try:
                if driver is not None:
                    driver.terminate()
            finally:
                if qlabs is not None:
                    qlabs.close()


def main() -> None:
    path = plan_path()
    print(f"Planned {len(path)} waypoints from {tuple(start)} to {tuple(goal)}.")
    print("RRT path:", np.round(np.asarray(path), 2))
    run_in_qlabs(path)


if __name__ == "__main__":
    main()
