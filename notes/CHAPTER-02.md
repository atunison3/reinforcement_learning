
# Chapter 2

- **Define Problem** - In "RL you must define the three core elements of the problem: the reward, the actions, and the environment".
- **Policy Evaluation** - an agent takes an actions $a$ from possible actions $\mathcal{A}$. Mathematically speaking this is expressed as the following. $$a\in\mathcal{A}$$
- The agent is rewarded with reward $r\in\mathcal{R}$.
- After rewarding, environment moves to a new state $s\in\mathcal{S}$.
- All of these variables are **stochastic** meaning, given the same input state and action $s$ and $a$, the resulting state and action may be different.
- **Reward Engineering** - Solves the problem of what reard signal should be given for a given state-action response.
- Use the average reward $r^{avg}$ for a certain number of events (for the stochastic natural of things).

$$
\begin{align}
r^{avg}(a)\dot{=}\frac{1}{N(a)}\sum_{i=1}^{N(a)}r(a)_i=\frac{r_1+r_2+...+t_{N(a)}}{N(a)}\tag{2-1}
\end{align}
$$

- Saving the $N$ number of transaction data can be costly. This can simply be done by updating in place.

$$
\begin{align*}
r^{avg}_N&\gets\frac{1}{N}\sum_{i=1}^Nr_i\\
r^{avg}_N&\gets r_{N-1}^{avg}+\frac{1}{N}\left(r_N-r_{N-1}^{avg}\right)\tag{2-2}
\end{align*}
$$

- This is very similar to the general form of an exponentially weighted moving average.

$$r=r+\alpha\left(r'-r\right)\tag{2-3}$$

- Transition of a model

$$
p(s',r|s,a)\tag{2-4}
$$

- Bandit algorithms are "dynamic" A/B tests in that they begin with equal probability, but shift the behavior as the results stream in. Think of a button click (which color generates more sales?). In A/B testing, the company is losing 50% of the different in profit by using the "worse" color 50% of the time. If instead, the bandit uses 50% initially, then slowly begins changing it to the optimal color, the company won't lose as much.

- With simple problems, the reward at each step is very well defined. However, when deciding, we also need to account for all future rewards.

## Policies

Policies are the strategies to pick an action.

### Discounted Rewards

G is the total expected rewards from a current step. Calculating future rewards is accomplished through iteration. A policy is to map states to actions [preferable with highest probability of return $p(a|s)$] and is denoted by $\pi$.

$$
G\dot=r+r'+r...+r_T\tag{2-6}
$$

$$
G\dot=r+\gamma r'+\gamma^2r''+...=\sum_{k=0}^T\gamma^kr_k\tag{2-7}
$$

### Random

Though not stated, Winder previously used the random policy. This policy used a random function ($\epsilon=1$) to always select an action given a state. While the agent eventually learned the best action, it never acted on that information.

### State-Value Functions

$$
V_\pi(s)\dot=\operatorname{E}_\pi[G|s]=\operatorname{E}_\pi\left[\sum_{k=0}^T\gamma^kr_k|s\right]\tag{2-8}
$$

### Action-Value Function

This function is very similar to equation 2-8 but it provides a further selection to the state and action. It allows for recording the future expected return of any action at any given state.

$$
Q_\pi(s,a)\dot=\operatorname{E}_\pi[G|s,a]=\operatorname{E}_\pi\left[\sum_{k=0}^T\gamma^kr_k|s,a\right]\tag{2-9}
$$

### Optimal Policies

An optimal policy is the part of a policy that returns the action at a state **with the highest expected reward**. This is performed by selecting the action from a action-value function with the highest reward.

$$
\begin{align}\tag{2-10}
V_*(s)\dot=&\argmax_{a_s\in\mathcal{A}(s)}\operatorname{E}_{\pi_*}(s,a_s)\\
&\argmax_{a_s\in\mathcal{A}(s)}\operatorname{E}_{\pi_*}[G|s,a_s]
\end{align}
$$

## Policy Generation

The most fundamental policy generation is that of the state-value function. This iterates through all possible episodes and possibilities of rewards. However, in most cases, this is impractical due to the sheer volume of possible trajectories.

### Monte Carlo

Performs a Monte Carlo simulation of episodes and trajectories.

### Value Iteration

An algorithm that looks ahead of the current state to see potential rewards. It then chooses an action based on the next step. This leads to policy stabilization starting at the terminal state and moving to the commence state.

- All code is available at [RL Book](https://rl-book.com)
- Winder's implementation of Bandit algorithms [gitlab](https://gitlab.com/WinderAI/rl/BanditsBook)
