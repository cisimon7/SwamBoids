from binaries import Simulation

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
    )

    simulation.flock()

    simulation.run(flock_size=150)
