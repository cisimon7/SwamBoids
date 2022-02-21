import numpy as np
from gymBoidEnv import new_simulation_env
from binaries import Flock, Boid, Vector2D, KDTree
from binaries import Simulation

if __name__ == '__main__':
    simulation = new_simulation_env()

    flock: Flock = simulation.flock  # Casting as Flock
    shapes = simulation.shapes

    FRAME_RATE = 60
    WINDOW_WIDTH = 1500
    WINDOW_HEIGHT = 900
    PERCEPTION = 100


    def on_each_frame():
        # Create KDTree structure for faster searching of nearby boids
        tree: KDTree = KDTree(WINDOW_WIDTH, WINDOW_HEIGHT)
        boids = flock.boids
        for _boid in boids:
            tree.insert(_boid)

        # Get the neighboring boids for each boid
        neighbors = list(map(
            lambda boid__: tree.search(boid__, PERCEPTION),
            flock.boids
        ))

        for (boid_, neighbors_) in zip(flock.boids, neighbors):
            boid_.update(neighbors_)


    # simulation.add_boid(1.0, 1.0, False)
    simulation.step_run(flock_size=10, pred_size=10, on_frame=(lambda: print("Hello Py")), delay_ms=300)
    # simulation.run(flock_size=10, pred_size=3, on_frame=on_each_frame, ext_update=False)
    # print(flock.boids[0])
    # boid: Boid = flock.boids[0]
    # pos: Vector2D = boid.position
    # print(pos)
    # print(boid.boid_id)
