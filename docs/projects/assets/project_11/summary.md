# Project 11 — Hex-world travel

Map seed: 42; unit seed: 42; turn limit: 500.
Training: epsilon=0.1, alpha=0.1, gamma=1; zero initial Q; distance potential shaping.
Learned aggregate state keys: 24903. Evaluation is greedy with frozen estimates, not proof of optimality.

## Training (20000 episodes)
- Successes: 19568
- Deaths: 432
- Timeouts: 0
- Mean successful travel time: 11.74 turns
- Mean unshaped return: 64.46

## Evaluation (100 episodes)
- Successes: 100
- Deaths: 0
- Timeouts: 0
- Mean successful travel time: 7.24 turns
- Mean unshaped return: 92.76

The SVGs show the start and trail of the first evaluation episode only.
No model weights, episode dataset, or large CSV are written.
