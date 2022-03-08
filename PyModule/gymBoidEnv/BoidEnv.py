from gym import Env
from .structs import *
from .boid_utils import *
from .reward_function import *
from typing import Optional, List
from datetime import datetime, timedelta
from ..binaries import Simulation, Flock, KDTree, new_simulation_env

RENDER_DELAY_MS = 300  # Delay during when rendering frame by frame
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

        self.start_time = datetime.now()
        self.evaluation_duration = timedelta(seconds=0, minutes=1)

        self.action_space = ActionSpace
        self.observation_space = BoidObsSpace

    def get_flock(self) -> Flock:
        """
        Retrieve flock from simulation instance. Implemented as a function instead of as a variable to get the current
        flock
        :return: current flock from simulation
        """
        return self.simulation.flock

    def set_flock(self, flock_: Flock):
        """
        Sets the flock in the simulation
        :param flock_:
        """
        self.simulation.flock = flock_

    def step(self, actions: BoidsAction) -> tuple[BoidsObservation, float, bool, dict]:
        """
        How new action changes current [observation, reward, done, info]
        :param actions:
        :return:
        """
        # Get Boids nearby to the main boid
        all_neighbors: List[list[Boid]] = self.all_neighbors()

        reward = 0
        observation_array = []
        for (boid, neighbors, action) in zip(self.simulation.flock.boids, all_neighbors, actions.reshape(-1, 2)):
            # retrieve observation vector from boid properties
            current_obs = obs_from_boid(boid)

            # Make decision based on current observation
            new_obs = obs_change(current_obs, neighbors, action)
            observation_array.append(new_obs)

            # update boid properties based on new observation and set it in world
            m_boid = update_boid(boid, new_obs)
            self.set_boid_by_id(m_boid.boid_id, m_boid)

            # calculate reward for single boid
            reward += calculate_reward(m_boid, neighbors)

        # done define is set to true after a period of time (evaluation_duration)
        done = ((datetime.now() - self.start_time).total_seconds()) > self.evaluation_duration.total_seconds()

        info = dict()
        return np.asarray(observation_array), reward, done, info

    def reset(self):
        self.simulation = new_simulation_env(FRAME_RATE)

        # Populate simulation with specified number of boids by calling step_run
        self.simulation.step_run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT,
                                 on_frame=(lambda: print("Initialized")), delay_ms=0, reset=True)

        # Returns initial Observation for a boid
        return self.observation_space.sample()

    def render(self, mode="human"):
        if mode != 'human':
            super(SwamBoidsEnv, self).render(mode=mode)  # Raises an exception

        if self.render_mode.value == RenderMode.EVALUATION.value:
            self.simulation.step_run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT,
                                     on_frame=(lambda: None), delay_ms=self.step_render_delay_ms, reset=False)
            # This can be set to use the initial flocking algorithm or use the trained model to update flocks
            # self.simulation.run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT, on_frame=(lambda: None))

    def flock_update_cpp(self):
        """
        Function to update flocks based on initial cpp flocking algorithm
        :return:
        """
        neighbors = self.all_neighbors()

        for (boid_, neighbors_) in zip(self.get_flock().boids, neighbors):
            boid_.update(neighbors_)  # TODO(To be changed)

    def all_neighbors(self) -> List[list[Boid]]:
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

    def boid_neighbors(self, boid_id: int) -> list[Boid]:
        # TODO(Since only considering nearest neighbors of one boid, is there a better algorithm than KDTree)
        # Create KDTree structure for faster searching of nearby boids
        tree: KDTree = KDTree(WINDOW_WIDTH, WINDOW_HEIGHT)
        boids: list[Boid] = self.get_flock().boids
        for boid in boids:
            tree.insert(boid)

        return tree.search(boids[boid_id], radius=PERCEPTION)

    def get_boid_by_id(self, id_) -> Boid:
        return list(filter(lambda boid: boid.boid_id == id_, self.get_flock().boids))[0]

    def set_boid_by_id(self, id_, new_boid: Boid):
        for (i, boid) in enumerate(self.simulation.flock.boids):
            if boid.boid_id == id_:
                self.simulation.flock.boids[i] = new_boid
