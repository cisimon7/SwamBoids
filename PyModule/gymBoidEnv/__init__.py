import logging

from gym import register

from PyModule.gymBoidEnv.BoidEnv import SwamBoidsEnv
from PyModule.gymBoidEnv.EnvStructs import ObsBoid, BoidAction

logger = logging.getLogger(__name__)

# Calling SwamBoidsEnv must call this function
register(
    id='SwamBoidsEnv-v0',
    entry_point='PyModule.gymEnv:SwamBoidsEnv'
)
