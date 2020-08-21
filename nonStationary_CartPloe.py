from __future__ import absolute_import, division, print_function
import threading
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
tf.compat.v1.enable_v2_behavior()

class Agent_Actor():
    def __init__(self,env_name,learning_rate,
                 num_episodes,replay_buffer_max_length,steps,
                 num_iterations,collect_step_per_iteration,
                 log_interval,batch_size,total_run_minutes,
                 perturb_duration,kernel_initializer):
        self.env_name = env_name
        train_py_env = suite_gym.load(self.env_name)
        eval_py_env = suite_gym.load(self.env_name)
        train_py_env.reset()
        eval_py_env.reset()
        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.replay_buffer_max_length = replay_buffer_max_length
        self.steps = steps
        self.num_iterations = num_iterations
        self.collect_step_per_iteration = collect_step_per_iteration
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.batch_size = batch_size
        self.returns = []
        self.total_run_minutes = total_run_minutes
        self.numLogs = 2*self.total_run_minutes
        self.numEvals = self.total_run_minutes
        self.eval_times=[]
        self.perturb_duration = perturb_duration
        self.kernel_initializer = kernel_initializer
        self.perturb_times = []

        self.agent_init()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.train_env.batch_size,
            max_length = self.replay_buffer_max_length
            )
        self.start_time = time.time()
        
        agent_thread = threading.Thread(target=self.train_agent)
        actor_thread = threading.Thread(target=self.act)
        
        actor_thread.start()
        agent_thread.start()

        actor_thread.join()
        agent_thread.join()

    def agent_init(self):
        ### the agent here is a DQN
        fc_layer_params = (100,)
        q_net = q_network.QNetwork(self.train_env.observation_spec(),
                                   self.train_env.action_spec(),
                                   kernel_initializer = self.kernel_initializer,
                                   fc_layer_params=fc_layer_params)

        ### define the agent
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step_counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer = optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        self.agent.initialize()

        ### define the policies
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.data_spec = self.agent.collect_data_spec

    def actor(self,perturb):
        time_step = self.train_env.current_time_step()
        action_step = self.collect_policy.action(time_step)

        if perturb:
            if action_step.action.numpy()[0] == 0:
                perturbed_action = tf.constant([1])
            else:
                perturbed_action = tf.constant([0])
            next_time_step = self.train_env.step(perturbed_action)
        else:
            next_time_step = self.train_env.step(action_step.action)

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

    def dataset_constructor(self):
        dataset = self.replay_buffer.as_dataset(num_parallel_calls=3,
                                                sample_batch_size=self.batch_size,
                                                num_steps=2,
                                                single_deterministic_pass=False)
        self.iterator = iter(dataset)

    def collect_data(self,perturb):
        for _ in range(self.steps):
            self.actor(perturb)

    def compute_avg_return(self,perturb):
        total_return = 0.0
        for _ in range(self.num_episodes):
            
            time_step = self.eval_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.eval_policy.action(time_step)
                if perturb:
                    if action_step.action.numpy()[0] == 0:
                        perturbed_action = tf.constant([1])
                    else:
                        perturbed_action = tf.constant([0])
                    time_step = self.eval_env.step(perturbed_action)
                else:
                    time_step = self.eval_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
            
        avg_return = total_return/self.num_episodes
        return avg_return
    def act(self):
        perturb = False
        last_perturbed = time.time()
        while time.time()-self.start_time < self.total_run_minutes*60:
            if time.time()-last_perturbed > self.perturb_duration*60:
                perturb = not perturb
                last_perturbed = time.time()
            self.collect_data(perturb)
        
    def train_agent(self):
        print("The agent starts training.")
        ### code optimization, wrapping the code  in a graph using TF functions
        self.agent.train = common.function(self.agent.train)

        ### reset the train step
        self.agent.train_step_counter.assign(0)

        ### evaluate the agent's policy once before trainig
        perturb = False
        avg_return = self.compute_avg_return(perturb)
        self.returns.append(avg_return)
        last_log_time = time.time()
        last_eval_time = time.time()
        last_perturbed = time.time()
        while time.time()-self.start_time < self.total_run_minutes*60:
            if time.time()-last_perturbed > self.perturb_duration*60:
                perturb = not perturb
                last_perturbed = time.time()
                self.perturb_times.append(time.time())
            
            ### construct a dataset from the replay buffer
            time_bfr_train = time.time()
            self.dataset_constructor()
            
            ### sample a batch of data from the buffer and update the agents network
            expereince, unused_info = next(self.iterator)
            train_loss = self.agent.train(expereince).loss
            step = self.agent.train_step_counter.numpy()
            

            if time.time() - last_log_time > self.total_run_minutes*60/self.numLogs:
                last_log_time = time.time()
                print('step = {0}: loss = {1} Elapsed time = {2}'.format(step, train_loss, time.time()-self.start_time))
            
            if time.time() - last_eval_time > self.total_run_minutes*60/self.numEvals:
                last_eval_time = time.time()
                avg_return = self.compute_avg_return(perturb)
                print('step = {0}: Average Return = {1} Elasped_time = {2}'.format(step, 0, time.time()-self.start_time))
                self.returns.append(avg_return)
                self.eval_times.append(last_eval_time-self.start_time)
            

def plot_returns(time_steps_vec,returns_vec,buffer_lengths,total_run_minutes,perturb_duration,perturb_times):
    
    colors = ['peru','dodgerblue','darkorchid']
    colors, buffer_lengths = iter(colors), iter(buffer_lengths)
    episode_mid_times = [x*60 for x in range(perturb_duration//2,total_run_minutes,perturb_duration)]
    for returns in returns_vec:
        evaluation_times, avg_returns = compute_eps_avg_ret(returns,total_run_minutes,perturb_times)
        plt.plot(evaluation_times,avg_returns,color=next(colors),linestyle='-',label='L=%d'%next(buffer_lengths))
    plt.legend(loc=0)
    plt.xlabel("Time")
    plt.ylabel("Average Return")
    plt.savefig('returns.pdf')
    plt.show()

def compute_eps_avg_ret(returns,total_run_minutes,perturb_times):
    print('this is returns:',returns)
    eval_averaging_interval = len(returns)//10
    eval_averaging_interval_minutes = total_run_minutes//10
    eps_avg_ret = []
    evaluation_times = []
    perturb_times = iter(perturb_times)
    i = 0

    for i in range(10):
        evaluation_times.append((i+1)*eval_averaging_interval_minutes)
        eps_avg_ret.append(sum(returns[i*eval_averaging_interval:(i+1)*eval_averaging_interval])/eval_averaging_interval)
    return evaluation_times, eps_avg_ret
        
def kernel_init():
    return tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03)

if __name__ == "__main__":
    ### hyperparameters
    learning_rate = 1e-3
    batch_size = 128 ### the number of observations (per experiment) used in an update.
    log_interval = 100 ### the interval for printing the train loss

    num_episodes = 100 ### the number of episodes an updated policy gets evaluated in an environment for computing average return
    steps = 100 ### the number of actions in the environment with an updated policy

    num_iterations = 20000 ### total number of policy updates

    collect_step_per_iteration = 1 
    initial_collect_steps = 1000
    replay_buffer_max_length = [200,20000,2000000]

    num_eval_episode = 100  ### the number of episodes an updated policy gets evaluated in an environment for computing average return
    eval_interval = 1000 ### the number of iterations before an evaluation takes place
    total_run_minutes = 90
    perturb_duration = 10
    ### Defining the environment
    env_name = 'CartPole-v0'
    returns = []
    time_steps = []
    perturb_times = []
    num_runs = 5
    kernel_initializer_vec = []
    for _ in range(num_runs):
        kernel_initializer_vec.append(kernel_init())
    for L in replay_buffer_max_length:
        returns_temp = [0 for _ in range(total_run_minutes-1)]
        time_steps_temp = [0 for _ in range(total_run_minutes-1)]
        perturb_times_temp = [0 for _ in range(total_run_minutes-1)]
        for run in range(num_runs):
            cart_pole = Agent_Actor(env_name,learning_rate,
                                    num_episodes,L,steps,
                                    num_iterations,collect_step_per_iteration,
                                    log_interval,batch_size,total_run_minutes,
                                    perturb_duration,kernel_initializer_vec[run])
            time_steps_temp = [x+y for (x,y) in zip(cart_pole.eval_times[:len(time_steps_temp)],time_steps_temp)]
            perturb_times_temp = [x+y for (x,y) in zip(cart_pole.perturb_times[:len(perturb_times_temp)],perturb_times_temp)]
            returns_temp = [x+y for (x,y) in zip(cart_pole.returns[:len(returns_temp)],returns_temp)]
        returns.append([x/num_runs for x in returns_temp])
        perturb_times.append([x/num_runs for x in perturb_times_temp])
        time_steps.append([x/num_runs for x in time_steps_temp])
    plot_returns(time_steps,returns,replay_buffer_max_length,total_run_minutes,perturb_duration,perturb_times)
    
