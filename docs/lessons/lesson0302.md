# Lesson 3.2 — Goals and Rewards

In [Lesson 3.1](lesson0301.md), the agent chose actions while the environment supplied states and rewards. Section 3.2 asks a different question: **what should those rewards mean?** They should make achieving a high return correspond to achieving the intended goal—not merely following a suggested strategy.

Source: `reinforcement_learning/lessons/lesson0302.py`. This lesson illustrates Sutton and Barto's discussion of goals and rewards; it is not a numbered textbook exercise response.

## 1. The reward hypothesis

The reward hypothesis proposes that goals and purposes can usefully be understood as maximizing the expected cumulative sum of a scalar reward signal. It is a hypothesis about how to formulate goals, not a theorem that every human preference has a convenient reward representation.

For this lesson, we use a finite episode of $T$ decisions, with no discounting:

$$
G_0 = \sum_{t=0}^{T-1} R_{t+1},
\qquad
J(\pi) = \mathbb{E}_{\pi}[G_0].
$$

- **Reward $R_{t+1}$:** the single numerical signal received after action $A_t$.
- **Return $G_0$:** the total reward accumulated during this episode.
- **Expected return $J(\pi)$:** the average return implied by a policy and the environment's randomness.

The objective is not to maximize each individual reward independently. A cost now can enable larger rewards later. Nor does a high expected return promise a good outcome on every episode: a policy that produces return $10$ with probability $0.3$ and $0$ otherwise has expected return $3$, which exceeds a guaranteed return of $2$. It still produces less than $2$ in most episodes.

The code below is deterministic, so each fixed policy's return equals its expected return. General continuing tasks require a suitable return definition, such as discounting or average reward; an infinite, undiscounted sum need not be finite.

## 2. Describe the outcome, not a recipe

Consider the examples in the section:

| Intended goal | Possible reward | What must be checked? |
| --- | --- | --- |
| Move forward | Reward proportional to forward displacement per step | Does motion remain safe and useful? Forward progress alone does not encode balance or energy limits. |
| Escape a maze quickly | $-1$ on every action up to and including the action that exits | For an escape taking $T$ actions, return is $-T$; maximizing expected return minimizes expected escape time when this expectation is finite. |
| Recycle cans | $+1$ for each genuinely collected/recycled can, zero otherwise | Can the same can be counted repeatedly? Do collisions or other costs need to be represented? |
| Win a board game | $+1$ for a win, $-1$ for a loss, and $0$ otherwise | Capturing pieces is useful only insofar as it improves the final outcome. |

For the board-game scheme, expected return is $\Pr(\text{win})-\Pr(\text{loss})$. When draws are possible, this is not identical to maximizing win probability alone. Reward choices specify preferences between outcomes, including draws.

**Why not reward taking chess pieces?** Taking pieces is a possible means to winning. If capturing a queen earns enough reward to outweigh subsequently losing, an agent can improve its return while worsening the outcome we actually wanted. The error is in the objective, not in the agent's arithmetic.

An RL method attempts to improve reward; it is not guaranteed to find an optimal policy. But even perfect optimization cannot repair a reward definition that values the wrong behavior.

## 3. Python example: recycling versus repeated pickups

Our intended goal is to recycle one can during **six decisions**. A pickup brings the can into the robot's hand; putting it down makes another pickup possible. Recycling removes it irreversibly. After recycling, only waiting is available until the six-decision horizon ends.

| Can status | Action | Next can status |
| --- | --- | --- |
| On floor | Pick up | Held |
| Held | Put down | On floor |
| Held | Recycle | Recycled |
| Any status | Wait | Unchanged |

The state also records the number of decisions remaining. At zero remaining decisions, the episode ends and no actions are available.

Compare two reward definitions over **the same physical transitions**:

1. **Outcome reward:** $+1$ when the can is actually recycled; $0$ otherwise.
2. **Proxy reward:** $+1$ every time the can is picked up; $0$ otherwise.

Then evaluate two fixed policies:

- `recycle_policy`: pick up, recycle, then wait.
- `repeat_pickup_policy`: alternate picking up and putting down the same can.

### Run the lesson

From the repository root, in the project's Python environment:

```bash
python -m reinforcement_learning.lessons.lesson0302
```

The program prints actions and rewards for all four combinations. Its totals are:

| Reward rule | Policy | Six rewards | Return | Cans recycled |
| --- | --- | --- | ---: | ---: |
| Actual recycling | Recycle | 0, 1, 0, 0, 0, 0 | 1 | 1 |
| Actual recycling | Repeat pickup | 0, 0, 0, 0, 0, 0 | 0 | 0 |
| Each pickup | Recycle | 1, 0, 0, 0, 0, 0 | 1 | 1 |
| Each pickup | Repeat pickup | 1, 0, 1, 0, 1, 0 | 3 | 0 |

**Interpretation.** Under the outcome reward, recycling ranks above repeated pickups. Under the proxy reward, the ranking reverses. Putting down the can gives no immediate reward, but it creates another opportunity to earn a pickup reward. The loophole therefore exploits cumulative reward, not just a greedy preference for the current reward.

The repeated-pickup policy achieves the maximum possible pickup return of $3$ in six decisions: each rewarded pickup after the first needs an intervening put-down. This does not mean that every pickup-optimal policy must fail to recycle. A policy can replace the final put-down with recycling and also obtain return $3$. The problem is that the proxy does **not distinguish** that successful outcome from leaving the can on the floor.

The actual-recycling reward aligns with the stated goal in this toy task, but does not prefer early recycling over late recycling within the horizon. Both produce return $1$. If speed matters, it must be part of the objective rather than assumed to follow from success alone.

### Walk through the code

```python
from reinforcement_learning.lessons.lesson0302 import (
    Can,
    RecyclingEnvironment,
    RewardRule,
    repeat_pickup_policy,
    run_episode,
)

environment = RecyclingEnvironment(RewardRule.PICKUP)
trajectory = run_episode(environment, repeat_pickup_policy)

reward_total = sum(transition.reward for transition in trajectory)
cans_recycled = int(trajectory[-1].next_state.can is Can.RECYCLED)
print(reward_total, cans_recycled)  # 3.0 0
```

In `RecyclingEnvironment.step`, the environment first applies the action's physical consequences. It then calculates reward according to its configured rule. The policy receives a state and chooses an action; it does not supply its own reward.

`run_episode` records transitions $(S_t,A_t,R_{t+1},S_{t+1})$. There are no parameter updates. **This example evaluates hand-written policies; it does not demonstrate learning, exploration, or convergence.** The separate `cans_recycled` measurement lets us compare the intended outcome against the numerical objective.

## 4. Questions raised by the section

### Is reward the same thing as the goal?

Reward is how the task designer formalizes the goal for the learning problem. The intended goal and its implementation can disagree. In the example, “recycle the can” is the intention, while “maximize pickups” is the goal actually induced by the proxy signal. A higher reward does not prove that the original intention was satisfied.

### Why is one scalar signal flexible enough to be useful?

A scalar can assign value to many kinds of outcomes: distance traveled, time spent, resources collected, or a final game result. It can also combine criteria, for example

$$
R_{t+1}=\text{new cans recycled}-c\,\text{collisions},
\qquad c>0.
$$

But this specifies a tradeoff, not a hard prohibition on collisions. An agent may accept a collision if the extra recycling reward outweighs its cost. Safety constraints or preferences that must never be traded away may require more than a casually chosen weighted sum. Scalar rewards are useful; choosing a faithful one is not automatic.

### Does a negative reward force the agent to stop immediately?

No. It must consider later consequences. In the maze example, a negative step reward discourages unnecessary delay, provided escaping is the relevant way to end the episode. If another action could terminate the episode without escape and avoid future costs, the reward and termination rules might instead encourage that shortcut.

For a guaranteed successful escape taking $T$ actions, the specified return is $-T$. The relation to expected escape time assumes policies for which escape occurs with finite expected duration. A fixed timeout with no extra failure treatment is a different task and should be checked separately.

### Why not reward every useful subgoal?

A subgoal is not necessarily the final goal. Our proxy pays for a step that usually helps recycling, but the robot can undo it and repeat it. Rewarding a correlated event changes the objective unless additional care is taken.

Prior knowledge can instead influence a model, representation, initialization, or planning procedure. There are also principled reward-shaping methods: for example, potential-based shaping can preserve optimal policies under suitable assumptions, including appropriate terminal handling. That is not a blanket justification for arbitrary subgoal bonuses. This lesson deliberately demonstrates a naive proxy, not such a policy-preserving construction.

### Why is reward external when hunger or battery charge is internal to a body?

“External” refers to the learning agent's control boundary, not the physical enclosure. A battery, hunger sensor, or reward computation can be inside a robot or organism while remaining outside the component that selects actions.

The agent should influence the outcome by acting—charging the battery, finding food, delivering the can—not simply declare that success occurred. In the script, the reward rule is configured by the experiment, not chosen by the policy. Python object boundaries illustrate this separation; they are not security barriers against arbitrary code modification.

### Can an agent create internal rewards?

Yes. A learning method can introduce an intrinsic signal, such as a novelty bonus, to guide exploration or help learn useful behavior. This does not make the original task reward arbitrary. The auxiliary signal must be distinguished from the external criterion used to judge success; maximizing an internal bonus alone may not solve the original task.

For example, novelty could encourage a recycling robot to inspect unexplored locations. Whether that helps should ultimately be assessed by actual recycling performance, not merely the novelty score. The script uses no internal reward.

### Does shifting all rewards by a constant leave the goal unchanged?

Not always. In this lesson, every episode has exactly six rewarded transitions, so adding $c$ to every transition reward adds the same $6c$ to every return and preserves rankings. With variable-length, undiscounted episodes, the added term is $cT$ and can favor longer or shorter episodes. Episode boundaries and return definitions matter alongside the reward numbers.

## 5. Check your understanding

1. **Why track cans recycled separately from return?** To detect disagreement between the intended outcome and the reward being optimized. Under pickup rewards, return $3$ can accompany zero recycled cans.
2. **Would increasing the pickup bonus fix the problem?** No. It would strengthen the incentive to repeat the same rewarded event without requiring recycling.
3. **What happens with a one-decision horizon?** Neither policy can recycle. Under the outcome rule both receive $0$; under the proxy both receive $1$. The task must allow enough time for the desired outcome.
4. **What happens with an eight-decision horizon?** Recycling still yields one recycled can. Repeated pickups yield proxy return $4$ and no recycling. More opportunities to collect proxy rewards do not repair the goal.
5. **Does the outcome rule require a particular method?** No. It rewards recycling, not an exact action sequence. The simple environment happens to offer few routes; a richer one could allow many successful methods.

The central lesson is: **specify what success means, then check whether maximizing the proposed reward could avoid that success.**
