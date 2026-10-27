"""QLabs scene setup: spawn the virtual PVC pipes."""
from qvl.basic_shape import QLabsBasicShape


def spawn_obstacles(qlabs, centers, r_pvc, height):
    """Spawn one cylinder per (x, y) in `centers` inside QLabs."""
    shapes = QLabsBasicShape(qlabs)
    scale = [r_pvc * 2, r_pvc * 2, height]  # scale is diameter, diameter, height

    for x, y in centers:
        shapes.spawn(
            location=[x, y, height / 2],
            rotation=[0, 0, 0],
            scale=scale,
            configuration=shapes.SHAPE_CYLINDER,
            waitForConfirmation=True,
        )
