
import numpy as np
import random
from ushiriki_policy_engine_library.SimpleChallengeEnvironment import ChallengeEnvironment
from ushiriki_policy_engine_library.EvaluateSubmission import EvaluateAugmentedChallengeSubmission,  EvaluateChallengeSubmission


class ChallengeEnvironment1(ChallengeEnvironment):
    def __init__(self):
        ChallengeEnvironment.__init__(
            self, baseuri="http://alpha-upe-challenge.eu-gb.mybluemix.net", experimentCount=2000)


class CustomAgent:
    def __init__(self, environment):
        self.environment = environment

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(20):
                self.environment.reset()
                policy = {}
                # episode length
                for j in range(self.environment.policyDimension):
                    policy[str(
                        j+1)] = [random.random(), random.random()]
                    candidates.append(policy)
                rewards = self.environment.evaluatePolicy(candidates)
                best_policy = candidates[np.argmax(rewards)]
                best_reward = rewards[np.argmax(rewards)]
        except(  KeyboardInterrupt, SystemExit ):
            print(exc_info())
        return best_policy, best_reward


eval = EvaluateAugmentedChallengeSubmission(
    ChallengeEnvironment1, CustomAgent, "test.csv")
