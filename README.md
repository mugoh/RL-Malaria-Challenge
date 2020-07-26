### Policy Search solution for the [Ushiriki Malaria Challenge(IBM Africa)](https://github.com/IBM/ushiriki-policy-engine-library)

This problem aims to seek the most optimal combination of action intervention methods to control the transmission, prevalence and health outcomes of malaria infection, should be distributed in a simulated human population. The solution using Reinforcement Learning, to identify sequential decisions that yield the largest accumulative reward for Malaria intervention measured in the population.


#### Environment Info
*State*

Observations occur over a 5 year timeframe. The `env` gives five states `[1, 2, 3, 4, 5]`. State transition is independent of action selected.


*Actions*

The action is a tuple of possible continous intervention methods `[0, 1]` i.e., Insecticide Spraying and Distributing Nets


*Reward*

The reward is a function of the received Health Outcomes of the policy-selected intervention measures per unit cost. *`-r(inf, inf)`*

#### Setup
1. Install the Ushiriki library
```
    $ pip install git+https://github.com/ibm/ushiriki-policy-engine-library --user user_name_cred
```

2. Install the package
```
    $ pip install -e .
```


#### Running and Evaluation
- To run the policy on the environment, use the command

```
    $ python3 scripts/run_ushiriki_psearch.py --env_name ushr --gae --ep_len 5 --discount .99 --n_iter 20 -rtg
```

`ep_len` is the length of the episode. In this particular env, a single episode lasts
for 5 years. This value should therefore be fixed

---

###### Other Params:
`--gae`: Whether use Generalised Advantage Estimates in estimating the returns


`--size`: Network size

`--n_iter`: Iterations to run the agent

`--batch_size`: Training batch size

`--eval_batch_size`: Evaluation batch size

`--discount`: Reward decay factor

`--rtg`: Compute reward-to-go



**NOTE**: This challenges was originally part of [Indaba19](https://zindi.africa/competitions/ibm-malaria-challenge) and credentials would be needed to evaluate the policy actions. i.e `userID` and `baseuri` (See more in the [policy engine library](https://github.com/IBM/ushiriki-policy-engine-library))

Credential params
`--userID`
`--baseuri`
