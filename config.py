"""Shared parameters for the QBot RRT project. Edit these to match your cage."""

# [x_min, x_max, y_min, y_max] in metres (measured with a tape measure)
CAGE_BOUNDS = (-2.0, 2.0, -2.0, 2.0)

# (x, y) of each PVC pipe, relative to the QBot start position
PVC_CENTERS = [
    (0.0, -1.0),
    (0.5, 0.4),
    (-0.7, 0.5),
    (0.0, 1.5),
    (0.0, 1.0),
]

R_PVC = 0.05      # physical radius of a PVC pipe (m)
R_QBOT = 0.17     # approximate radius of the QBot (m)
PVC_HEIGHT = 0.60 # pipe height (m), used only for the QLabs visual

START = (0.0, 0.0)
GOAL = (1.5, 1.5)  # change to wherever you want the QBot to end up
