from typing import Tuple
from gym import Env, register
from EnvStructs import BoidAction, BoidObs
from ..binaries import Simulation, Flock, Boid, KDTree

FRAME_RATE = 60


class SwamBoidsEnv(Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': FRAME_RATE
    }

    def __init__(self):
        self.simulation = new_simulation_env(FRAME_RATE)

    def get_flock(self) -> Flock: return self.simulation.flock

    def set_flock(self, flock: Flock): self.simulation.flock = flock

    def step(self, action: BoidAction) -> Tuple[BoidObs, float, bool, dict]:
        """
            Run one time step of the environment's dynamics.
            Args:
                action (object): an action provided by the agent
            Returns:
                observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (bool): whether the episode has ended. if true, step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information
        """
        # TODO(
        #   Action updates BoidObs and also boid in flock
        #   Step controls frame render frame rate
        #   Update simulation
        # )
        # TODO(How should cohesion, alignment and separation affect reward)
        # TODO(Done is currently set to false as there is no end to the life of the agent - boid)
        info = dict()
        return BoidObs(), 0.0, False, info

    def reset(self):
        self.simulation = new_simulation_env(FRAME_RATE)
        return BoidObs()

    def render(self, mode="human"):
        if mode != 'human':
            super(SwamBoidsEnv, self).render(mode=mode)  # just raise an exception

        # TODO(Pass flock to simulation, call sfml clear screen and call on draw for new boid positions)
        self.simulation.run(
            step_render=self.step_render,
            flock_size=0
        )

    def step_render(self):
        pass


register(
    id='SwamBoidsEnv-v0',
    entry_point='PyModule.gymEnv:SwamBoidsEnv'
)


def new_simulation_env(frame_rate: int):
    return Simulation(
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
        frame_rate=frame_rate
    )
