# Chapter 3 Temporal-Difference Learning, Q-Learning, and n-Step Algorithms

- *Temporal-Difference* (TD) - combination of dyanmic programming (DP) and bootstrapping.
- For updating expected return of a decision for online applications, we can use the online Monte Carlo state-value function.

$$
\text{3-1}\qquad
\begin{align*}
V_\pi(s)\dot=&\operatorname{E}_\pi[G|s]\\
&\leftarrow\operatorname{E}_\pi\left[V_\pi(s)+\alpha\left(G-V_\pi(s)\right)|s\right]
\end{align*}
$$


