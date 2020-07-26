import multiprocessing
import concurrent.futures
import time

from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os

from ushiriki.infrastructure.utils import *
from ushiriki.infrastructure.tf_utils import create_tf_session
from ushiriki.infrastructure.logger import Logger

from .custom_ushiriki_env import CustomUshirikiEnvironment


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])
        env_creds = self.params['env_creds']
        self.sess = create_tf_session(
            self.params['use_gpu'], which_gpu=self.params['which_gpu'])

        # Set random seeds
        seed = self.params['seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #############
        # ENV
        #############

        # Make the gym environment
        # self.env = gym.make(self.params['env_name'])
        # self.env.seed(seed)
        self.env = CustomUshirikiEnvironment(**env_creds)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or \
            self.env.policyDimension
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        # discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        discrete = False
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        # ob_dim = self.env.observation_space.shape[0]
        ob_dim = self.env.observation_dim
        # ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        ac_dim = self.env.actionDimension
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata['video.frames_per_second']

        #############
        # AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(
            self.sess, self.env, self.params['agent_params'])

        #############
        # INIT VARS
        #############

        tf.global_variables_initializer().run(session=self.sess)

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        if self.params['parallel']:
            batch_s = self.params['batch_size']
            cores = multiprocessing.cpu_count()
            batch_per_core = batch_s // cores
            rem = batch_s % cores
            batches = [batch_per_core] * cores

            if rem:
                batches[:rem] = batches[:rem] + 1
            print(f'Starting threading: using {cores} cores')

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            self.training_loss = []
            self.val_loss = []

            # collect trajectories, to be used for training
            training_returns = []

            if self.params['parallel']:

                with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
                    future = {
                        executor.submit(
                            self.collect_training_trajectories, itr, initial_expertdata, collect_policy, b_s):
                        b_s for b_s in batches
                    }

                    for trajectory in concurrent.futures.as_completed(future):
                        try:
                            data = trajectory.result()
                        except Exception as e:
                            print(f'Generated exception : {e}')
                        else:
                            training_returns.append(data)

                returns = np.asanyarray(training_returns)
                training_returns = np.concatenate(returns[:,
                                                          0]), returns[:, 1].sum(), returns[:, 2]
                if np.any(training_returns[2]):
                    training_returns[2] = np.concatenate(training_returns[2])

            else:
                training_returns = self.collect_training_trajectories(itr,
                                                                      initial_expertdata, collect_policy,
                                                                      self.params['batch_size'])

            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                if 'ushiriki' in self.params['env_name'].lower():
                    self.log_ushiriki(eval_policy)
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths)

                if self.params['save_params']:
                    # save policy
                    print('\nSaving agent\'s actor...')
                    self.agent.actor.save(
                        self.params['logdir'] + '/policy_itr_'+str(itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy, batch_size):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """


        if not itr and load_initial_expertdata:
            with open(load_initial_expertdata, 'rb') as f:
                initial_expert_data = pickle.load(f)
            return initial_expert_data, 0, None
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(
            self.env, collect_policy, batch_size, max_path_length=self.params['ep_len'])

        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            # : look in utils and implement sample_n_trajectories
            train_video_paths = sample_n_trajectories(
                self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return [paths, envsteps_this_batch, train_video_paths]

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            sampled_data = self.agent.sample(self.params['train_batch_size'])

            steps = self.params['multistep']

            print('\n == Using multistep PG ==') if steps > 1 else None

            for step in range(steps):
                print(f'\nmultistep: {step}\n') if not step % 2 else None

                loss = self.agent.train(*sampled_data)
                if isinstance(loss, tuple):
                    train_loss, val_loss = loss
                    self.training_loss += [train_loss]
                    self.val_loss += [val_loss]
                else:
                    self.training_loss += [loss]

                # print(f'loss {loss}')

    def do_relabel_with_expert(self, expert_policy, paths):

        print("\nRelabelling collected observations with labels from an expert policy...")

        for path in paths:
            path['action'] = expert_policy.get_action(path['observation'])

        return paths

        ####################################
        ####################################

    def log_ushiriki(self, eval_policy):
        """
            Evaluate on Unshiriki env
        """
        self.best_rews = []
        candidates = []
        for step in range(self.params['eval_ep_lens']):
            ob = self.env.reset()
            policy = {}

            for i in range(self.env.policyDimension):
                policy[step + i] = eval_policy.get_action(ob)[0]
                candidates.append(policy)
            rew = self.env.evaluatePolicy(candidates)
            best_idx = np.armax(rew)
            best_policy = candidates[best_idx]
            best_rew = rew[best_idx]
        self.best_rews.append(best_rew)

    def perform_logging(self, itr, paths, eval_policy, train_video_paths):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(
            self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(
                self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum()
                            for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"])
                            for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs['Training_loss_Average'] = np.mean(self.training_loss)
            logs["Best_Ushiriki_Eval_policy_mean"] = np.mean(self.best_rews)

            if self.val_loss:
                logs['Value_loss_Average'] = np.mean(self.val_loss)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
