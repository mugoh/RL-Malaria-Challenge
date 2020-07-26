Results of experiment runs using different implementations of policy gradient
## Problem 3: Cartpole
- Policy Gradient experiment on discrete CartPole-v0 environment

#### Commands
```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name sb_no_rtg_dsa --n_layers 4
```

```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name sb_rtg_dsa --n_layers 4
```

```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name sb_rtg_na --n_layers 4
```

```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name lb_no_rtg_dsa --n_layers 4
```

```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name lb_rtg_dsa --n_layers 4
```

```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name lb_rtg_na --n_layers 4
```

### Deliverables
![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex3%20Cartpole/lbs.png)
*Large batches (5000) at different lr*

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex3%20Cartpole/sbs.png)
*Small batches (1000)*

#### Generating the image plots
In the hw2 directory, run the command
`python3 generate_plot.py 3`



##### Questions
**a)** Which value estimator has better performance without advantage-standardization: the trajectory-centric one, or the one using reward-to-go?


![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex3%20Cartpole/Small_batch_rtg_vs_no_rtg_Screenshot%20from%202020-04-25%2013-36-59.png)

- The estimator using reward to go(rtg) convrges more quickly and has a more stable return/learning curve after the 80th step. Applying causality (rtg) minimizes variance

**b)** Did advantage standardization help?


![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex3%20Cartpole/Small_batch_dsa_vs_nodsaScreenshot%20from%202020-04-25%2013-51-19.png)
- Apparently. Standardization seems to improve learning stability but lowers the speed

**Note**: In CArtpole, The agent receives a reward with a value of 1 for every step it survives. So the rewards are in a consistent range of values, and standardizing them seems safe, as there aren't important rare events with extremely high rewards.

However, this may not generalize well in other environments[different reward values]
see [this question](https://ai.stackexchange.com/questions/10196/why-does-is-make-sense-to-normalize-rewards-per-episode-in-reinforcement-learnin)

My interpretation of this is standardizing advantages would similary scale down states of higher value in environments with large variance on the advantage of sampled states.

**c)** Did the batch size make an impact?


*Yes*

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex3%20Cartpole/Small_batch_vs_large_batch_rtg_adv_standardizationScreenshot%20from%202020-04-25%2013-55-55.png)
- Using a larger batch size results  in lower variance in the model during learning. The graph shows the learning is quicker and more stable for the large_batch compared to the small batch


## Problem 4 : InvertedPendulum

> Finding smallest batch size and largest learning rate that gets to optimum in less that 100 iterations
##### Command
`$ export BATCH_SIZE=5000`
`$ export LR=2e-2`

```
python run_hw2_policy_gradient.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $BATCH_SIZE -lr $LR -rtg --exp_name ip_b"$BATCH_SIZE"_r"$LR"
```

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex4%20Inverted%20Pendulum/optimum_bs_lr2.png)
*perfomance at bs=5000 and lr=.02*

#### Generating the image plots
In the hw2 directory, run the command
`python3 generate_plot.py 4`




## Problem 6: LunarLadar
> task: Test Baseline implementation

> **Expectation**: Average return of around 180

##### command
```
python run_hw2_policy_gradient.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005
```

#### Deliverable
![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex6%20lunar%20ladar/problem_6_lunar_ladar.png)
*LunarLadar learning curve*


## Problem 7: HalfCheeter

> Search over batch sizes [10000, 30000, 50000] and learning rates [0.005, 0.01, 0.02]

##### command
```
export BATCH_SIZE=50000 && export LR=.02 && python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH_SIZE -lr $LR --video_log_freq -1 --reward_to_go --nn_baseline --exp_name hc_b"$BATCH_SIZE"_lr"$LR"_nnbaseline
```

#### Deliverables:
![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex7%20HalfCheeter/half_cheeter_blr_search.png)

- There's a steady increase in the average returns with increase in the batch size for the same number of steps, with increased training time.
- Increase in learning rate results to faster learning[higher rewards in same learning time]. However, this is also dependant on the batch size - Smaller batches working better with smaller lr(The b10,000 did best on a lr.01, while b30,000 and b50,000 on a lr.02)


##### Commands
`export BATCH_SIZE=30000 && export LR=.02`

```
 python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH_SIZE -lr $LR --video_log_freq -1 --exp_name hc_b"$BATCH_SIZE"_r"$LR"
```

```
python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH_SIZE -lr $LR --video_log_freq -1 -rtg --exp_name hc_b"$BATCH_SIZE"_r"$LR"
```

```
python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH_SIZE -lr $LR --video_log_freq -1 --nn_baseline --exp_name hc_b"$BATCH_SIZE"_r"$LR"
```

```
python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH_SIZE -lr $LR --video_log_freq -1 -rtg --nn_baseline --exp_name hc_b"$BATCH_SIZE"_r"$LR"
```

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/ex7%20HalfCheeter/rtg_baseline.png)


## Experiment 9
### Parallelization of trajectory collection

`export BATCH_SIZE=100000 && export LR=.02`

```
python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH_SIZE -lr $LR --video_log_freq -1 -rtg --exp_name parr_hc_b"$BATCH_SIZE"_r"$LR" --parallel
```

- The training time is faster when parallelized when using very large batches. For smaller batches (< 10,000), collecting trajectories on a single thread achieves higher reward averages for smaller time steps compared to the parallelized collection process.

### Generalized Advantage Estimation (GAE)

No GAE
```
python run_hw2_policy_gradient.py --env_name Walker2d-v2 --ep_len 1000 --discount 0.995 -n 100 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --nn_baseline --exp_name no_gae_1000_r0.005 
```

GAE
```
python run_hw2_policy_gradient.py --env_name Walker2d-v2 --ep_len 1000 --discount 0.995 -n 100 -l 2 -s 64 -b 1000 -lr 0.005 -rtg --nn_baseline --exp_name gae_1000_r0.005 --lambda 0.95 --gae
```

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/experiment%209/gae_no_gae.png)


### Multistep Policy Gradient

Single-step pg
```
python run_hw2_policy_gradient.py --env_name Walker2d-v2 --discount 0.99 -n 100 -l 2 -s 64 -b 1000 -lr 0.01 -rtg --nn_baseline --exp_name no_multistep_b1000_r0.01
```

200
```
python run_hw2_policy_gradient.py --env_name Walker2d-v2 --discount 0.99 -n 100 -l 2 -s 64 -b 1000 -lr 0.01 -rtg --nn_baseline --multistep 3 --exp_name multistep3_b1000_r0.01
```

500
```
python run_hw2_policy_gradient.py --env_name Walker2d-v2 --discount 0.99 -n 100 -l 2 -s 64 -b 1000 -lr 0.01 -rtg --nn_baseline --multistep 5 --exp_name multistep5_b1000_r0.01
```

1000
```
python run_hw2_policy_gradient.py --env_name Walker2d-v2 --discount 0.99 -n 100 -l 2 -s 64 -b 1000 -lr 0.01 -rtg --nn_baseline --multistep 10 --exp_name multistep10_b1000_r0.01
```
- Doing multiple gradient updates with the same batch of updates affects perfomance drastically. The agent doesn't seem to learn at all. **Explanation for that?**

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/experiment%209/multistep_vs_singlestep_pg.png)
*Multi-step vs single-step. update steps: 3, 5, 10*

![alt text](https://github.com/mugoh/deepRL-cs285/blob/master/hw2/cs285/.figures_csv/experiment%209/multistep_half_cheeter.png)
*Multistep vs single-step half-cheeter. update step: 3*


