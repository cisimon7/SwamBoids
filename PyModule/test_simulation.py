from binaries import Flock, KDTree
from gymBoidEnv import new_simulation_env

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

        # print(neighbors)

        for (boid_, neighbors_) in zip(flock.boids, neighbors):
            boid_.update(neighbors_)


    while True:
        simulation.step_run(flock_size=100, pred_size=10, on_frame=(lambda: on_each_frame()), delay_ms=20, reset=False)
