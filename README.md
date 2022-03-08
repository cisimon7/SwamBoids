### Ongoing Research project:

Python3 version used: 3.9

### Running code:

The above python version was used to when compiling the cpp wrappers.
To use the available wrappers in the repository, it is best to use the same python version.

A virtual environment was created and the wrappers were saved in the virtual environment with directory: "PyModule/venv/lib/python3.9/site-packages". This was done to be able to get autocompletion working in CLion IDE

You can change the location by setting the LIBRARY_DIR flag in the CMakeList.txt file on line 8 to any location you want, but you would have to also reset the import directory in the "PyModule/binaries/__init__.py".
Build the c++ code to generate the binaries in your desired location, then run the python code.

### Project Description:

The flocking behavior of swarms of birds can be modelled by creating a swarm of artificial births (boids) which react to observations by using simple control rules that increase cohesion,
alignment and separation. Research in swarm robotics is attempting to design swarms which can perform additional functions such as coordinately exploring an unknown area to identify
people to be rescued.

In this project we will evolve large swarms of boids for the ability to stay away from moving predators. Each boid is provided with a camera from which it can extract the relative position
and orientation of nearby boids and the relative positions of nearby obstacles. Moreover, each boid will be provided with a speaker used to emit sounds and a microphone used to detect the
sound signals produced by nearby boids. The neural network brain of boids will be adapted by using the OpenAI-ES evolutionary algorithm or the PPO reinforcement learning algorithm. 

The scientific objective of this project is to verify whether evolving boids can develop an ability to coordinate and cooperate to avoid long distant dangers and eventually an ability to monitor a
large space region that enables them to identify dangers as quickly as possible.

* Adaptive Algorithm: Reinforcement learning (PPO)
* Library: Stable-baselines3 and OpenAI gym


### Adding New Algorithm

To add a new algorithm to be used for PPO, simply edit the definition for the "calculate_reward" function found in the directory "PyModule/gymBoidEnv/reward_function.py"
