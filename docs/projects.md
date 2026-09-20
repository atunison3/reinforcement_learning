# Projects

Personal reinforcement-learning experiments and things I build live in `reinforcement_learning/projects/`. Project numbers are local identifiers, not Sutton and Barto exercise numbers. A project may draw inspiration from a book exercise without being filed as a book response.

| Project | Topic |
| --- | --- |
| [1](documentation/project001.md) | Two-action cliff/safe bandit |
| [2](documentation/project002.md) | Gaussian bandit and epsilon-greedy selection ([results](projects/project2.md)) |
| [3](projects/project3.md) | Tracking nonstationary rewards |
| [4](projects/project4.md) | Optimistic initial values |
| [5](projects/project5.md) | UCB versus epsilon-greedy |
| [6](projects/project6.md) | Gradient bandits |
| [7](projects/project7.md) | Associative search |
| [8](projects/project8.md) | Blackjack parameter study |
| [9](projects/project9.md) | Gradient-bandit reward baselines |
| [10](projects/project10.md) | Full-round blackjack, splitting, and card counting |

Run a project from the repository root, for example:

```bash
python -m reinforcement_learning.projects.project007
```

Generated figures and CSVs live under `docs/projects/assets/`. Actual textbook responses belong in [Exercises](exercises/README.md), and concept-focused material belongs in [Lessons](lessons.md).
