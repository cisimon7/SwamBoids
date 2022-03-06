import logging

from gym.spaces import MultiDiscrete, MultiBinary
import numpy as np
from numpy.random import rand
from typing import Tuple, List, Optional, Set, Dict

from gym import Env
from gym.envs.registration import register

from ..binaries import Simulation, Flock, KDTree, Boid, Vector2D
from ..gymBoidEnv.Structs import RenderMode, np_vector2D, createObs, ObsvBoid, split, ActionBoid
from ..config import FRAME_RATE, WINDOW_WIDTH, WINDOW_HEIGHT, PERCEPTION
from ..config import MAX_SPEED, MAX_FORCE, ALIGNMENT_WEIGHT, COHESION_WEIGHT, SEPARATION_WEIGHT, ACCELERATION_SCALE, \
    SEPARATION_DISTANCE, NOISE_SCALE, BOID_SIZE, NUM_THREADS, BOID_COUNT

RENDER_DELAY_MS = 300  # Delay during when rendering frame by frame
BOID_COUNT_ = BOID_COUNT  # Initial boid count
PREDATOR_COUNT = 5  # Initial predator count
UNIT_STEP = 1  # Unit step of boid per frame change


class SwamBoidsEnv(Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': FRAME_RATE
    }

    def __init__(self):
        self.render_mode: RenderMode = RenderMode.TRAINING
        self.step_render_delay_ms = RENDER_DELAY_MS
        self.simulation: Optional[Simulation] = None

        # The main boid is the boid being trained in the simulation environment
        self.main_boid_id: Optional[int] = None  # To be set upon reset environment

        self.action_space = ActionBoid
        self.observation_space = ObsvBoid

    def get_flock(self) -> Flock:
        return self.simulation.flock

    def set_flock(self, flock_: Flock):
        self.simulation.flock = flock_

    def step(self, action: ActionBoid) -> Tuple[np.ndarray, float, bool, dict]:
        # Get Boids nearby to the main boid
        neighbors: List[Boid] = self.boid_neighbors(self.main_boid_id)

        m_boid = self.get_boid_by_id(self.main_boid_id)

        pos = np_vector2D(m_boid.position)  # Pos.from_vector2D(m_boid.position)
        vel = np_vector2D(m_boid.velocity)  # Vel.from_vector2D(m_boid.velocity)
        acc = np_vector2D(m_boid.acceleration)  # Acc.from_vector2D(m_boid.acceleration)

        # current observation for main boid
        current_obs = np.c_[pos, vel, acc].ravel()

        # Make decision based on current observation
        new_obs = obs_change(current_obs, neighbors, action)
        (n_pos, n_vel, n_acc) = split(new_obs)

        # update main boid
        c_pos = self.pos_const(n_pos)  # Toroidal world
        m_boid.position = Vector2D(c_pos[0], c_pos[1])
        m_boid.velocity = Vector2D(n_vel[0], n_vel[1])
        m_boid.acceleration = Vector2D(n_acc[0], n_acc[1])
        self.set_boid_by_id(m_boid.boid_id, m_boid)

        # update other boids that are not the main boid
        self.update_other_bids()

        # calculate reward based on new observation
        reward = 0.0

        # define done
        done = False

        info = dict(prev_neighbors=neighbors)
        return new_obs, reward, done, info

    def reset(self):
        self.simulation = new_simulation_env(FRAME_RATE)

        # Populate simulation with specified number of boids by calling step_run
        self.simulation.step_run(flock_size=BOID_COUNT_, pred_size=PREDATOR_COUNT,
                                 on_frame=(lambda: print("Initialized")), delay_ms=0)
        self.main_boid_id = self.simulation.flock.boids[0].boid_id

        return self.observation_space.sample()  # Returns initial Observation for a boid

    def render(self, mode="human"):
        if mode != 'human':
            super(SwamBoidsEnv, self).render(mode=mode)  # Raises an exception

        if self.render_mode.value is RenderMode.TRAINING.value:
            self.simulation.step_run(flock_size=BOID_COUNT_, pred_size=PREDATOR_COUNT,
                                     on_frame=(lambda: None), delay_ms=self.step_render_delay_ms)
        else:
            # This simply uses the flocking algorithm
            self.simulation.run(flock_size=BOID_COUNT_, pred_size=PREDATOR_COUNT, on_frame=(lambda: None))

    def flock_update_cpp(self):
        # update flocks based on initial cpp algorithm
        neighbors = self.all_neighbors()

        for (boid_, neighbors_) in zip(self.get_flock().boids, neighbors):
            boid_.update(neighbors_)  # TODO(To be changed)

    def all_neighbors(self) -> List[List[Boid]]:
        # Create KDTree structure for faster searching of nearby boids
        tree: KDTree = KDTree(WINDOW_WIDTH, WINDOW_HEIGHT)
        boids = self.get_flock().boids
        for boid in boids:
            tree.insert(boid)

        # Get the neighboring boids for each boid
        neighbors = list(map(
            lambda boid__: tree.search(boid__, PERCEPTION),
            self.get_flock().boids
        ))
        return neighbors

    def boid_neighbors(self, boid_id: int) -> List[Boid]:
        # TODO(Since only considering nearest neighbors of one boid, is there a better algorithm than KDTree)
        # Create KDTree structure for faster searching of nearby boids
        tree: KDTree = KDTree(WINDOW_WIDTH, WINDOW_HEIGHT)
        boids: List[Boid] = self.get_flock().boids
        for boid in boids:
            tree.insert(boid)

        return tree.search(boids[boid_id], radius=PERCEPTION)

    def pos_const(self, pos) -> np.ndarray:
        # Taking into account the toroidal universe
        if pos[0] < 0: pos[0] += WINDOW_WIDTH
        if pos[1] < 0: pos[1] += WINDOW_HEIGHT
        if pos[0] >= WINDOW_WIDTH: pos[0] -= WINDOW_WIDTH
        if pos[1] >= WINDOW_HEIGHT: pos[1] -= WINDOW_HEIGHT
        return pos

    def update_other_bids(self):
        boids: List[Boid] = self.get_flock().boids
        for boid in boids:
            if boid.boid_id != self.main_boid_id:  # main boid already updated by training
                pos = self.pos_const(np.array([boid.position.x + UNIT_STEP, boid.position.y]))  # Toroidal universe
                boid.position = Vector2D(pos[0], pos[1])

    def get_boid_by_id(self, id_) -> Boid:
        return list(filter(lambda boid: boid.boid_id == id_, self.get_flock().boids))[0]

    def set_boid_by_id(self, id_, new_boid: Boid):
        for (i, boid) in enumerate(self.simulation.flock.boids):
            if boid.boid_id == id_:
                self.simulation.flock.boids[i] = new_boid


def new_simulation_env(frame_rate: int = FRAME_RATE) -> Simulation:
    return Simulation(
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        boid_size=BOID_SIZE,
        max_speed=MAX_SPEED,
        max_force=MAX_FORCE,
        alignment_weight=ALIGNMENT_WEIGHT,
        cohesion_weight=COHESION_WEIGHT,
        separation_weight=SEPARATION_WEIGHT,
        acceleration_scale=ACCELERATION_SCALE,
        perception=PERCEPTION,
        separation_distance=SEPARATION_DISTANCE,
        noise_scale=NOISE_SCALE,
        fullscreen=False,
        light_scheme=True,
        num_threads=NUM_THREADS,
        frame_rate=frame_rate
    )


def obs_change(obs: np.ndarray, neighbors: List[Boid], action: ActionBoid) -> np.ndarray:
    # How action should change observation
    return np.c_[obs[:2] + action, obs[2:4], obs[4:6]].ravel()


logger = logging.getLogger(__name__)

# Calling SwamBoidsEnv must call this function
register(
    id='SwamBoidsEnv-v0',
    entry_point='PyModule.gymBoidEnv:SwamBoidsEnv'
)
