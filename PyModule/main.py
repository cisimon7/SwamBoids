from binaries import Simulation
from binaries import Flock

if __name__ == '__main__':
    simulation = Simulation(
        window_width=1500,
        window_height=900,
        boid_size=4,
        max_speed=6,
        max_force=1,
        alignment_weight=0.65,
        cohesion_weight=0.75,
        separation_weight=4.5,
        acceleration_scale=0.3,
        perception=100,
        separation_distance=20,
        noise_scale=0,
        fullscreen=False,
        light_scheme=True,
        num_threads=-1,
        frame_rate=60
    )

    flock: Flock = simulation.flock  # Casting as Flock
    shapes = simulation.shapes

    print(shapes)


    def on_each_frame():
        # print("Hello Python")
        if flock.size() > 100:
            flock.clear()
            simulation.shapes = []
        print(flock.size())


    simulation.run(flock_size=10, on_frame=on_each_frame)
