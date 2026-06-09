
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


- Bandit algorithms are "dynamic" A/B tests in that they begin with equal probability, but shift the behavior as the results stream in. Think of a button click (which color generates more sales?). In A/B testing, the company is losing 50% of the different in profit by using the "worse" color 50% of the time. If instead, the bandit uses 50% initially, then slowly begins changing it to the optimal color, the company won't lose as much.

- With simple problems, the reward at each step is very well defined. However, when deciding, we also need to account for all future rewards.

$$
G\dot=r+r'+r...+r_T\tag{2-6}
$$

$$
G\dot=r+\gamma r'+\gamma^2r''+...=\sum_{k=0}^T\gamma^kr_k\tag{2-7}
$$

- All code is available at [RL Book](https://rl-book.com)
