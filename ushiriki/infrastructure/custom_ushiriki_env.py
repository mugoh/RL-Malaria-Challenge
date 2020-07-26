"""
    This custom env should add bares compatibility for common gym env calls
"""
from ushiriki_policy_engine_library.DLI19ChallengeEnvironment \
    import ChallengeEnvironment
from dataclasses import dataclass
import numpy as np


@dataclass
class EnvData:
    metadata = {'video.frames_per_second': 50}


class CustomUshirikiEnvironment(ChallengeEnvironment):
    """
        Custome Ushiriki Env intended for compatibility with
        common Gym env method calls
    """

    def __init__(self,
                 baseuri="http://alpha-upe-challenge.eu-gb.mybluemix.net",
                 **args):
        super(CustomUshirikiEnvironment, self).__init__(baseuri, **args)
        self.observation_dim = 1
        self.env = EnvData

    def reset(self):
        """
            Reset initial state
        """
        super().reset()
        obs = 1
        return np.array([obs])

    def step(self, ac):
        """Take action step"""
        return super().evaluateAction(ac)
