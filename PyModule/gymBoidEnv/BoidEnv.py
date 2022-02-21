from gym import Env
from typing import Tuple
from ..gymBoidEnv.Structs import ActionBoid, ObsBoid, rnd_obs
from ..binaries import Simulation, Flock, Boid, KDTree

import logging
from gym.envs.registration import register

FRAME_RATE = 60


class SwamBoidsEnv(Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': FRAME_RATE
    }

    def __init__(self):
        self.step_render = True
        self.simulation = new_simulation_env(FRAME_RATE)
        self.flock = self.simulation.flock
        self.shapes = self.simulation.shapes

    def get_flock(self) -> Flock: return self.simulation.flock

    def set_flock(self, flock: Flock): self.simulation.flock = flock

    def step(self, action: ActionBoid) -> Tuple[ObsBoid, float, bool, dict]:
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
        return rnd_obs, 0.0, False, info

    def reset(self):
        self.simulation = new_simulation_env(FRAME_RATE)
        return rnd_obs  # Returns initial Observation for a boid

    def render(self, mode="human"):
        if mode != 'human':
            super(SwamBoidsEnv, self).render(mode=mode)  # just raise an exception

        if self.step_render:
            self.simulation.step_run(flock_size=10, on_frame=(lambda: None))
        else:
            self.simulation.run(flock_size=10, on_frame=(lambda: None))

    def step_update(self):
        # TODO(Update simulation flocks)
        self.flock = Flock()
        self.shapes = []


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


logger = logging.getLogger(__name__)

# Calling SwamBoidsEnv must call this function
register(
    id='SwamBoidsEnv-v0',
    entry_point='PyModule.gymBoidEnv:SwamBoidsEnv'
)
