# Exercise 3

## Tracking a Nonstationary Problem

Most of the theory of action-value methods is developed under the assumption that the problem is *stationary*—that the true values \(q_*(a)\) do not change over time. That assumption is often false. In many of the applications that matter, the environment drifts, and what was best yesterday need not be best today. The question is then not merely how to *find* the optimal action, but how to *track* it.

Exercise 3 makes this concrete with a variant of the ten-armed testbed. At the start of each run all true action values are equal (zero). After every time step, each \(q_*(a)\) takes an independent random walk:

\[
q_*(a) \;\leftarrow\; q_*(a) + \mathcal{N}(0,\,0.01).
\]

The reward actually received on selecting action \(a\) is still noisy, \(R_t \sim \mathcal{N}\bigl(q_*(a),\,1\bigr)\), but now the means themselves are nonstationary, and the identity of \(a^*_t = \arg\max_a q_*(a)\) wanders. An \(\varepsilon\)-greedy agent with \(\varepsilon = 0.1\) must continually reassess its estimates \(Q_t(a)\).

We compare two step-size rules for the familiar incremental update

\[
Q_{n+1} = Q_n + \alpha_n\bigl[R_n - Q_n\bigr].
\]

The first is the sample-average choice \(\alpha_n = 1/n\), which is guaranteed to converge to the true value when the problem is stationary. The second holds the step-size parameter fixed at a constant \(\alpha = 0.1\). Performance is assessed over 2,000 independent runs of 10,000 steps each, in terms of average reward and the percentage of times the optimal action is selected—the same measures used for the stationary testbed, now applied to a problem that will not sit still.

## Why a Constant Step Size?

The sample-average method is elegant, but it is designed for a world that does not change. As \(n\) grows, \(\alpha_n = 1/n\) shrinks toward zero. Eventually new rewards barely move \(Q_n\); the estimate has effectively frozen. That is desirable when \(q_*(a)\) is fixed—one wants the variance of the estimate to vanish—but it is a liability when \(q_*(a)\) is a moving target. Old data become misleading, yet they continue to dominate the average.

A constant step size \(\alpha \in (0,1]\) avoids this freeze. Expanding the recurrence yields an *exponentially weighted* average of past rewards and the initial estimate:

$$
\newcommand{\rn}[1]{\alpha R_{n#1}}
\newcommand{\qn}[1]{\rn{#1} + (1 - \alpha)Q_{n#1}}
\begin{align*}
Q_{n+1}&=Q_n+\alpha[R_n-Q_n]\\
&=Q_n+\alpha R_n -\alpha Q_n\\
&=\rn{} + Q_n - \alpha Q_n\\
&=\qn{}\\
&=\rn{} + (1 - \alpha)\left[\qn{-1}\right]\\
&=\rn{} + (1 - \alpha)\rn{-1} + (1 - \alpha)^2Q_{n-1}\\
&=\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ...(1-\alpha)^{n-1}\rn{-n+1} + (1 - \alpha)^nQ_1\\
&=(1-\alpha)^nQ_1+\alpha\sum_{i=1}^n(1-\alpha)^{n-i}R_i.
\end{align*}
$$

The weight on reward \(R_i\) is \(\alpha(1-\alpha)^{n-i}\). Recent rewards are weighted most heavily; the influence of the distant past decays geometrically. One sometimes says that the estimate has a *recency bias*, but that bias is exactly what tracking requires: the step-size parameter never goes to zero, so \(Q_n\) remains plastic enough to follow a slow drift such as our \(\mathcal{N}(0,0.01)\) walk.

Of course \(\alpha\) must still be chosen with care. If it is too large, the estimate chatters with the noise in \(R_t\); if it is too small, tracking lags behind the true change in \(q_*(a)\). There is no single correct value for all problems—only a step-size parameter matched to the time scale on which the world actually changes. In that modest fact lies much of the art of reinforcement learning in nonstationary environments.

## Results

Average reward over 2,000 runs (sample average vs.\ \(\alpha = 0.1\)):

![Average Reward](assets/exercise_3_average_reward.png)

Percentage of optimal actions over the same runs:

![Percent Optimal Action](assets/exercise_3_optimal_action_percentage.png)
