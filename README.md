# ReinforcementLearning_DQN

This is a specific implementaion of [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) in [CartPole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) environment.
The added features are:
* The training of neural network and the experience collection are placed on *two seperate threads*.
* The cart-pole environment is *non-stationary*.

The implementation is based on [this tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb).
