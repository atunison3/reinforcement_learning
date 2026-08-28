# Exercise 3

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
&=(1-\alpha)^nQ_1+\alpha\sum_{i=1}^n(1-\alpha)^{n-i}R_i
\end{align*}
$$

Average reward across trials for &alpha; ∈ `{0.0, 0.1}` (default script settings: 2000 trials, 1000 steps):

![Average Reward](assets/exercise_3_average_reward.png)

Percent of optimal actions selected over the same runs:

![Percent Optimal Action](assets/exercise_3_optimal_action_percentage.png)