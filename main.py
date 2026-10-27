"""Plan a collision-free path for the QBot with RRT.

Usage:
    python main.py            # spawn QBot + pipes in QLabs, then plan
    python main.py --no-sim   # planner only (no Quanser libraries needed)
"""
import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from rrt import RRT


def plot_result(planner, path, out_file):
    fig, ax = plt.subplots(figsize=(6, 6))

    x_min, x_max, y_min, y_max = cfg.CAGE_BOUNDS
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min], "k-", label="Cage")

    for i, (cx, cy) in enumerate(cfg.PVC_CENTERS):
        ax.add_patch(plt.Circle((cx, cy), planner.inflated_radius,
                                color="tab:red", alpha=0.15,
                                label="Inflated obstacle" if i == 0 else None))
        ax.add_patch(plt.Circle((cx, cy), cfg.R_PVC, color="tab:red",
                                label="PVC pipe" if i == 0 else None))

    for node, parent in zip(planner.nodes, planner.parents):
        if parent != -1:
            p = planner.nodes[parent]
            ax.plot([p[0], node[0]], [p[1], node[1]], color="0.8", lw=0.8)

    if path:
        px, py = zip(*path)
        ax.plot(px, py, "b-", lw=2.5, label="RRT path")
    ax.plot(*cfg.START, "go", ms=9, label="Start")
    ax.plot(*cfg.GOAL, "m*", ms=14, label="Goal")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("QBot RRT path planning")
    ax.legend(loc="upper left", fontsize=8)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_file}")


def plan_path():
    planner = RRT(
        start=cfg.START, goal=cfg.GOAL, bounds=cfg.CAGE_BOUNDS,
        obstacles=cfg.PVC_CENTERS, obstacle_radius=cfg.R_PVC,
        robot_radius=cfg.R_QBOT, step_size=0.2, max_iter=5000, seed=42,
    )
    path = planner.plan()
    if path is None:
        print("No path found. Try more iterations or check the goal location.")
    else:
        print(f"Path found: {len(path)} waypoints, "
              f"{RRT.path_length(path):.2f} m long")
    return planner, path


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class QLabsQBot:
    def __init__(self, driver, start):
        self.driver = driver
        self.x, self.y = map(float, start)
        self.heading = 0.0
        self.started_at = time.time()
        self.last_update = None

    def pose(self):
        return self.x, self.y, self.heading

    def drive(self, forward_speed, turn_rate, duration):
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

    def stop(self):
        self.drive(0.0, 0.0, 0.1)


def wait_for_driver(driver, started_at, timeout=10.0):
    deadline = time.monotonic() + timeout
    zero_commands = np.zeros(2, dtype=np.float64)

    while time.monotonic() < deadline:
        if driver.read_write_std(
                timestamp=time.time() - started_at,
                arm=0,
                commands=zero_commands):
            return
        time.sleep(0.05)

    raise ConnectionError(
        "No data received from the QBot Platform driver at localhost. "
        "Check that QLabs is running and the virtual QBot driver is active."
    )


def follow_waypoints(robot, path, max_steps_per_waypoint=1200):
    max_forward_speed = 0.3
    max_turn_rate = 1.5

    for index, waypoint in enumerate(path[1:], start=1):
        target = np.asarray(waypoint)
        tolerance = 0.05 if index == len(path) - 1 else 0.12

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
                f"QBot did not reach waypoint {index}: "
                f"{np.round(target, 2)}"
            )

    robot.stop()


def run_with_sim():
    from qlabs_setup import setup
    from pal.products.qbot_platform import QBotPlatformDriver
    from qvl.basic_shape import QLabsBasicShape
    from qvl.qlabs import QuanserInteractiveLabs

    qlabs = my_qbot = robot = None
    try:
        setup(
            locationQBotP=[cfg.START[0], cfg.START[1], 0.05],
            rotationQBotP=[0, 0, 0],
            verbose=True,
        )
        time.sleep(2)

        qlabs = QuanserInteractiveLabs()
        if not qlabs.open("localhost"):
            raise ConnectionError("Could not connect to QLabs at localhost.")

        shape_spawner = QLabsBasicShape(qlabs)
        pvc_scale = [cfg.R_PVC * 2, cfg.R_PVC * 2, cfg.PVC_HEIGHT]
        print("Spawning virtual obstacles...")
        for x, y in cfg.PVC_CENTERS:
            shape_spawner.spawn(
                location=[x, y, cfg.PVC_HEIGHT / 2],
                rotation=[0, 0, 0],
                scale=pvc_scale,
                configuration=shape_spawner.SHAPE_CYLINDER,
                waitForConfirmation=True,
            )

        print("Connecting to virtual QBot...")
        my_qbot = QBotPlatformDriver(mode=1, ip="localhost")

        planner, path = plan_path()
        plot_result(planner, path, "media/rrt_path.png")
        if path is None:
            raise RuntimeError("Cannot drive the QBot because no path was found.")

        robot = QLabsQBot(my_qbot, cfg.START)
        wait_for_driver(my_qbot, robot.started_at)
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
                if my_qbot is not None:
                    my_qbot.terminate()
            finally:
                if qlabs is not None:
                    qlabs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sim", action="store_true",
                        help="run the planner only, without QLabs")
    args = parser.parse_args()

    if args.no_sim:
        planner, path = plan_path()
        plot_result(planner, path, "media/rrt_path.png")
    else:
        run_with_sim()
