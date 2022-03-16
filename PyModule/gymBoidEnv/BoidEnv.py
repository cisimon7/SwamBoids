import os

from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from .structs import *
from .boid_utils import *
from .reward_function import *
from typing import Optional, List, Callable
from datetime import datetime, timedelta
from ..config import FRAME_RATE, BOID_COUNT, PREDATOR_COUNT
from ..binaries import Simulation, Flock, KDTree, new_simulation_env

RENDER_DELAY_MS = 300  # Delay during when rendering frame by frame
UNIT_STEP = 1  # Unit step of boid per frame change


class SwamBoidsEnv(Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': FRAME_RATE
    }

    def __init__(self, best_model_dir=None):
        self.best_model_filename = None
        self.render_mode: RenderMode = RenderMode.TRAINING
        self.step_render_delay_ms = RENDER_DELAY_MS
        self.simulation: Optional[Simulation] = None

        self.start_time = datetime.now()
        self.evaluation_duration = timedelta(seconds=0, minutes=1)

        self.action_space = ActionSpace
        self.observation_space = BoidObsSpace
        self.reward_range = (-10, 0)

        self.main_boid_id = None
        self.best_model: Optional[BaseAlgorithm] = None
        self.best_model_dir = best_model_dir

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
        # main boid
        m_boid = self.get_boid_by_id(self.main_boid_id)

        # Get Boids nearby to the main boid
        neighbor: list[Boid] = self.boid_neighbors(self.main_boid_id)
        cur_obs = obs_from_boid(m_boid)
        new_obs = obs_change(cur_obs, neighbor, actions)

        m_boid = update_boid(m_boid, new_obs)
        self.set_boid_by_id(m_boid.boid_id, m_boid)

        reward = calculate_reward(m_boid, neighbor)
        reward += self.update_non_training_boids()

        # done define is set to true after a period of time (evaluation_duration)
        done = ((datetime.now() - self.start_time).total_seconds()) > self.evaluation_duration.total_seconds()

        return new_obs, reward, done, dict()

    def reset(self):
        self.simulation = new_simulation_env(FRAME_RATE)

        # Populate simulation with specified number of boids by calling step_run
        self.simulation.step_run(flock_size=BOID_COUNT, pred_size=PREDATOR_COUNT,
                                 on_frame=(lambda: print("Initialized")), delay_ms=0, reset=True)

        self.main_boid_id = self.simulation.flock.boids[0].boid_id
        for boid_ in self.simulation.flock.boids:
            boid_.is_predator = False

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

        neighbors = tree.search(boids[boid_id], radius=PERCEPTION)
        return list(filter(lambda boid_: boid_.boid_id != boid_id, neighbors))

    def get_boid_by_id(self, id_) -> Boid:
        return list(filter(lambda boid: boid.boid_id == id_, self.get_flock().boids))[0]

    def set_boid_by_id(self, id_, new_boid: Boid):
        for (i, boid) in enumerate(self.simulation.flock.boids):
            if boid.boid_id == id_:
                self.simulation.flock.boids[i] = new_boid

    def on_each_frame(self):
        # Create KDTree structure for faster searching of nearby boids
        tree: KDTree = KDTree(WINDOW_WIDTH, WINDOW_HEIGHT)
        boids = self.get_flock().boids
        for _boid in boids:
            tree.insert(_boid)

        # Get the neighboring boids for each boid
        neighbors = list(map(
            lambda boid__: tree.search(boid__, PERCEPTION),
            boids
        ))

        for (boid_, neighbors_) in zip(boids, neighbors):
            if boid_.boid_id != self.main_boid_id:
                boid_.update(neighbors_)

    def update_non_training_boids(self) -> float:
        others_reward = 0
        self.set_current_best_model()
        for (boid_, neighbors_) in zip(self.get_flock().boids, self.all_neighbors()):
            if boid_.boid_id != self.main_boid_id:  # Only update other boids
                cur_obs = obs_from_boid(boid_)
                if self.best_model is None:
                    action = self.action_space.sample()
                else:
                    action, _ = self.best_model.predict(cur_obs)[0]

                new_obs = obs_change(cur_obs, neighbors_, action)
                m_boid = update_boid(boid_, new_obs)
                self.set_boid_by_id(boid_.boid_id, m_boid)

                others_reward += calculate_reward(boid_, neighbors_)

        return others_reward

    def set_current_best_model(self):
        # load model if it's there
        model_list = [f for f in os.listdir(self.best_model_dir) if f.startswith("agent")]
        if len(model_list) > 0:

            if len(model_list) == 1:
                latest_model_filename = model_list[0]
            else:
                model_list.sort(key=lambda name: self.strip_rwd(name))
                latest_model_filename = model_list[-1]

            if self.best_model_filename is None:
                update = True
            else:
                best_reward = self.strip_rwd(self.best_model_filename)
                latest_best_reward = self.strip_rwd(latest_model_filename)
                update = latest_best_reward > best_reward

            if update:
                filename = os.path.join(self.best_model_dir, model_list[-1])  # the latest best model
                print("loading model: ", filename)
                if self.best_model is not None:
                    del self.best_model
                self.best_model = PPO.load(filename, env=self)  # load new main best
                self.best_model_filename = filename

    def strip_rwd(self, filename) -> float:
        name = filename.replace("agent:", "").replace(".zip", "").replace(self.best_model_dir, "").replace("/", "")
        return float(name)
