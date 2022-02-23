import logging
from numpy.random import rand
from typing import Tuple, List, Optional

from gym import Env
from gym.envs.registration import register

from ..binaries import Simulation, Flock, KDTree, Boid, Vector2D
from ..gymBoidEnv.Structs import ActionBoid, ObsBoid, rnd_obs, RenderMode, Pos, Vel, Acc
from ..config import FRAME_RATE, WINDOW_WIDTH, WINDOW_HEIGHT, PERCEPTION
from ..config import MAX_SPEED, MAX_FORCE, ALIGNMENT_WEIGHT, COHESION_WEIGHT, SEPARATION_WEIGHT, ACCELERATION_SCALE, \
    SEPARATION_DISTANCE, NOISE_SCALE, BOID_SIZE, NUM_THREADS

RENDER_DELAY_MS = 300  # Delay during when rendering frame by frame
BOID_COUNT = 50  # Initial boid count
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

    def get_flock(self) -> Flock:
        return self.simulation.flock

    def set_flock(self, flock_: Flock):
        self.simulation.flock = flock_

    def step(self, action: ActionBoid) -> Tuple[ObsBoid, float, bool, dict]:
        # Get Boids nearby to the main boid
        neighbors: List[Boid] = self.boid_neighbors(self.main_boid_id)

        m_boid = self.get_boid_by_id(self.main_boid_id)

        pos = Pos.from_vector2D(m_boid.position)
        vel = Vel.from_vector2D(m_boid.velocity)
        acc = Acc.from_vector2D(m_boid.acceleration)

        # current observation for main boid
        current_obs: ObsBoid = (pos, vel, acc, neighbors)

        # Make decision based on current observation
        new_obs = obs_change(current_obs, action)
        (n_pos, n_vel, n_acc, _) = new_obs

        # update main boid
        c_pos = self.pos_const(n_pos)  # Toroidal world
        m_boid.position = Vector2D(c_pos.x, c_pos.y)
        m_boid.velocity = Vector2D(n_vel.x, n_vel.y)
        m_boid.acceleration = Vector2D(n_acc.x, n_acc.y)
        self.set_boid_by_id(m_boid.boid_id, m_boid)

        # update other boids that are not the main boid
        self.update_other_bids()

        # calculate reward based on new observation
        reward = 0.0

        # define done
        done = False

        info = dict()
        return new_obs, reward, done, info

    def reset(self):
        self.simulation = new_simulation_env(FRAME_RATE)

        # Populate simulation with specified number of boids by calling step_run
        self.simulation.step_run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT,
                                 on_frame=(lambda: print("Initialized")), delay_ms=0)
        self.main_boid_id = self.simulation.flock.boids[0].boid_id

        return rnd_obs  # Returns initial Observation for a boid

    def render(self, mode="human"):
        if mode != 'human':
            super(SwamBoidsEnv, self).render(mode=mode)  # Raises an exception

        if self.render_mode.value is RenderMode.TRAINING.value:
            self.simulation.step_run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT,
                                     on_frame=(lambda: None), delay_ms=self.step_render_delay_ms)
        else:
            # This simply uses the flocking algorithm
            self.simulation.run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT, on_frame=(lambda: None))

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

    def pos_const(self, pos: Pos) -> Pos:
        # Taking into account the toroidal universe
        if pos.x < 0: pos.x += WINDOW_WIDTH
        if pos.y < 0: pos.y += WINDOW_HEIGHT
        if pos.x >= WINDOW_WIDTH: pos.x -= WINDOW_WIDTH
        if pos.y >= WINDOW_HEIGHT: pos.y -= WINDOW_HEIGHT
        return pos

    def update_other_bids(self):
        boids: List[Boid] = self.get_flock().boids
        for boid in boids:
            if boid.boid_id != self.main_boid_id:  # main boid already updated by training
                pos = self.pos_const(Pos(boid.position.x + UNIT_STEP, boid.position.y))  # Toroidal universe
                boid.position = Vector2D(pos.x, pos.y)

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


def obs_change(obs: ObsBoid, action: ActionBoid) -> ObsBoid:
    # How action should change observation
    return (
        obs[0] + action,
        obs[1],
        obs[2],
        obs[3]
    )


logger = logging.getLogger(__name__)

# Calling SwamBoidsEnv must call this function
register(
    id='SwamBoidsEnv-v0',
    entry_point='PyModule.gymBoidEnv:SwamBoidsEnv'
)
