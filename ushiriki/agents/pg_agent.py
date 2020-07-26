import numpy as np

from .base_agent import BaseAgent
from ushiriki.policies.MLP_policy import MLPPolicyPG
from ushiriki.infrastructure.replay_buffer import ReplayBuffer
from ushiriki.infrastructure.utils import *


class PGAgent(BaseAgent):
    def __init__(self, sess, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.sess = sess
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae = self.agent_params.get('gae')
        self.lamda = self.agent_params['lambda']
        # actor/policy
        # NOTICE that we are using MLPPolicyPG (hw2), instead of MLPPolicySL (hw1)
        # which indicates similar network structure (layout/inputs/outputs),
        # but differences in training procedure
        # between supervised learning and policy gradients
        self.actor = MLPPolicyPG(sess,
                                 self.agent_params['ac_dim'],
                                 self.agent_params['ob_dim'],
                                 self.agent_params['n_layers'],
                                 self.agent_params['size'],
                                 discrete=self.agent_params['discrete'],
                                 learning_rate=self.agent_params['learning_rate'],
                                 nn_baseline=self.agent_params['nn_baseline'],
                                 gae=self.agent_params.get('gae', False)
                                 )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, obs, acs, rews_list, next_obs, terminals):
        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.

            ----------------------------------------------------------------------------------

            Recall that the expression for the policy gradient PG is

                PG = E_{tau} [sum_{t=0}^{T-1} grad log pi(a_t|s_t) * (Q_t - b_t )]

                where
                tau=(s_0, a_0, s_1, a_1, s_2, a_2, ...) is a trajectory,
                Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                b_t is a baseline which may depend on s_t,
                and (Q_t - b_t ) is the advantage.

            Thus, the PG update performed by the actor needs (s_t, a_t, q_t, adv_t),
                and that is exactly what this function provides.

            ----------------------------------------------------------------------------------
        """

        if self.gae:
            q_values, advantage_values = self.use_gae(
                np.concatenate(rews_list), obs, terminals)

        else:
            # step 1: calculate q values of each (s_t, a_t) point,
            # using rewards from that full rollout of length T: (r_0, ..., r_t, ..., r_{T-1})
            q_values = self.calculate_q_vals(rews_list)

            # step 2: calculate advantages that correspond to each (s_t, a_t) point
            advantage_values = self.estimate_advantage(obs, q_values)

        loss = self.actor.update(
            obs, acs, qvals=q_values, adv_n=advantage_values)
        return loss

    def calculate_q_vals(self, rews_list):
        """
            Monte Carlo estimation of the Q function.

            arguments:
                rews_list: length: number of sampled rollouts
                    Each element corresponds to a particular rollout,
                    and contains an array of the rewards for every step of that particular rollout

            returns:
                q_values: shape: (sum/total number of steps across the rollouts)
                    Each entry corresponds to the estimated q(s_t,a_t) value
                    of the corresponding obs/ac point at time t.

        """

        # Case 1: trajectory-based PG
        if not self.reward_to_go:

            q_values = np.concatenate(
                [self._discounted_return(r) for r in rews_list])

        # Case 2: reward-to-go PG
        else:

            q_values = np.concatenate(
                [self._discounted_cumsum(r) for r in rews_list])

        return q_values

    def use_gae(self, rewards, obs, terminals):
        """
            GAE: Produces a more accurate estimate of the discounted advantage

            delta[t]: reward + V [t+1] - V[t]
            Adv = sigma[l=0: inf]([gamma * lambda] ^l * delta[t+1])
        """
        v_baseline = self.actor.run_baseline_prediction(obs)
        rew_len = rewards.size
        adv = np.zeros((rew_len,))

        for t in reversed(range(rew_len)):
            """
            delta = rewards[t] + ((1 - terminals[t]) *
                                  self.gamma * v_baseline[t+1]) - v_baseline[t]
            adv[t] = self.gamma * self.lamda * adv[t+1] + delta

            q_values = adv + v_baseline

            return q_values, adv """

            if terminals[t]:
                delta = rewards[t] - v_baseline[t]
                adv[t] = delta
            else:
                delta = rewards[t] + (1 - terminals[t]) * \
                    self.gamma * v_baseline[t+1] - v_baseline[t]

                adv[t] = delta + self.gamma * self.lamda * adv[t+1]
        q_values = adv + v_baseline

        return q_values, adv

    def estimate_advantage(self, obs, q_values):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """

        if self.nn_baseline:
            b_n_unnormalized = self.actor.run_baseline_prediction(obs)
            b_n = b_n_unnormalized * np.std(q_values) + np.mean(q_values)
            adv_n = q_values - b_n

        # Else, just set the advantage to [Q]
        else:
            adv_n = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return adv_n

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^{T-1} gamma^t' r_{t'}
                note that all entries of this output are equivalent
                because each index t is a sum from 0 to T-1 (and doesnt involve t)
        """

        # 1) create a list of indices (t'): from 0 to T-1
        rew_len = len(rewards)

        indices = np.arange(rew_len)

        # 2) create a list where the entry at each index (t') is gamma^(t')
        discounts = np.power(self.gamma, indices)

        # 3) create a list where the entry at each index (t') is gamma^(t') * r_{t'}
        discounted_rewards = np.multiply(discounts, rewards)

        # 4) calculate a scalar: sum_{t'=0}^{T-1} gamma^(t') * r_{t'}
        sum_of_discounted_rewards = np.sum(discounted_rewards)

        # 5) create a list of length T-1, where each entry t contains that scalar
        # NOTE why entries are all similar
        list_of_discounted_returns = np.full(
            rew_len, sum_of_discounted_rewards)

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Input:
                a list of length T
                a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T
            Output:
                a list of length T
                a list where the entry in each index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """

        all_discounted_cumsums = []
        rew_len = len(rewards)

        # for loop over steps (t) of the given rollout
        for start_time_index in range(len(rewards)):

            # 1) create a list of indices (t'): goes from t to T-1
            indices = np.arange(start_time_index, rew_len)

            # 2) create a list where the entry at each index (t') is gamma^(t'-t)
            discounts = np.power(self.gamma, indices - start_time_index)

            # 3) create a list where the entry at each index (t') is gamma^(t'-t) * r_{t'}
            # Hint: remember that t' goes from t to T-1, so you should use the rewards from those indices as well
            # np.multiply faster with arrays
            discounted_rtg = np.multiply(
                discounts, rewards[start_time_index:, ])

            # 4) calculate a scalar: sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
            sum_discounted_rtg = np.sum(discounted_rtg)

            # appending each of these calculated sums into the list to return
            all_discounted_cumsums.append(sum_discounted_rtg)
        list_of_discounted_cumsums = np.asanyarray(all_discounted_cumsums)
        return list_of_discounted_cumsums
