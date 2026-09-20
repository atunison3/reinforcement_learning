# Lesson 3.1 — The Agent–Environment Interface

The central idea in Sutton and Barto, Section 3.1, is a division of responsibility: **the agent chooses actions; the environment determines their consequences and the rewards that define the task.** Reinforcement learning concerns improving those choices through experience.

Source: `reinforcement_learning/lessons/lesson0301.py`. This is a concept lesson, not a numbered textbook exercise response.

## 1. Three signals and their timing

At decision time $t$, the agent receives $S_t$ and selects an available action $A_t \in \mathcal{A}(S_t)$. The environment then produces reward $R_{t+1}$ and next state $S_{t+1}$:

$$
S_t \;\xrightarrow{\text{agent chooses } A_t}\; (R_{t+1}, S_{t+1}).
$$

A trajectory is therefore

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, \ldots
$$

**Why is the reward indexed $t+1$?** It arrives as part of the response to $A_t$, not as a reward the agent chooses alongside that action. An initial state is needed before the first decision; this example does not invent an initial reward $R_0$.

The policy is a probability distribution over available actions:

$$
\pi_t(a\mid s) = \Pr(A_t=a\mid S_t=s),
\qquad \sum_{a\in\mathcal{A}(s)}\pi_t(a\mid s)=1.
$$

A deterministic policy assigns probability one to a single action. A stochastic policy can assign positive probability to several. The subscript $t$ allows the policy to change through learning; it does not require a change after every step.

## 2. Example: a delivery robot

A robot begins at a depot with one delivery opportunity. It can serve a nearby customer immediately or spend one decision stage traveling toward a distant customer.

| Current state | Available action | Next state | Reward | Episode ends? |
| --- | --- | --- | ---: | --- |
| Depot | Deliver nearby | Done | +2 | Yes |
| Depot | Drive toward distant customer | Road | −1 | No |
| Road | Finish distant delivery | Done | +5 | Yes |

For this small task, the goal is **maximum total reward for one delivery episode, without discounting**. Starting location, termination rules, available actions, transitions, and rewards are all part of the task specification. The numbers represent task-defined utility, not necessarily money.

- **Agent:** the component selecting a delivery action from the current state.
- **Environment:** the delivery process, including movement, customers, and reward calculation.
- **State:** `DEPOT`, `ROAD`, or `DONE`.
- **Actions:** delivery decisions, rather than individual motor voltages.
- **Reward:** an externally specified numerical assessment of each transition.

`ROAD` includes the fact that the distant delivery is still pending. In this deliberately simple model, it contains everything needed to determine the remaining consequences. Real delivery tasks would usually need more information.

### Run the lesson

From the repository root, using the project's Python environment:

```bash
python -m reinforcement_learning.lessons.lesson0301
```

The essential output is:

```text
Immediate-reward policy:
S_0=depot; A_0=deliver nearby; R_1=+2; S_1=done
Total episode reward: 2

Delayed-reward policy:
S_0=depot; A_0=drive toward distant customer; R_1=-1; S_1=road
S_1=road; A_1=finish distant delivery; R_2=+5; S_2=done
Total episode reward: 4
```

**What does this demonstrate?** Selecting the action with the highest immediate reward gives $2$. Accepting the initial cost gives $-1+5=4$. Reward is a signal at each transition; the objective concerns reward accumulated over time. The best immediate action need not be the best action for the task.

This comparison depends on the stated objective. With a discount factor $\gamma$, the distant route's return would be $-1+5\gamma$ and would exceed $2$ only when $\gamma>0.6$. A continuing reward-per-unit-time task could give another comparison. Section 3.1 motivates long-term reward; it does not make these different objectives interchangeable.

### Walk through the Python interface

```python
from reinforcement_learning.lessons.lesson0301 import DeliveryEnvironment, PolicyAgent

environment = DeliveryEnvironment()
agent = PolicyAgent(probability_far=1.0)

state = environment.reset()             # S_0
while True:
    action = agent.choose_action(state)  # A_t, sampled from pi(a | S_t)
    outcome = environment.step(action)  # R_{t+1}, S_{t+1}
    # A learning algorithm could update its policy from this experience.
    state = outcome.next_state
    if outcome.terminated:
        break
```

`Outcome` bundles the next state, reward, and termination flag. The flag is an implementation convenience for stopping the episode, not a fourth goal-defining signal. The terminal state has no available actions; the environment rejects actions after termination.

Try `PolicyAgent(probability_far=0.25, seed=7)`. At the depot, its policy assigns probability $0.25$ to driving far and $0.75$ to delivering nearby. Its expected episode reward is

$$
0.75(2)+0.25(4)=2.5.
$$

A single episode still yields either $2$ or $4$, not $2.5$. The expectation summarizes repeated episodes. On the road, the only available action has probability one.

**Does the script learn?** No. It compares fixed policies to isolate the interface. Sampling actions randomly is not itself learning. A learning implementation would use experience to change action probabilities or the estimates that determine them. Here the probability is explicitly set by the reader.

## 3. Questions raised by the section

### Must one time step be one second?

No. A time step identifies a decision stage. Driving to a customer could take minutes while choosing a route takes milliseconds. This lesson treats each transition as one stage and does not optimize elapsed time. If duration affects the objective, it must be modeled rather than silently ignored.

### Is the robot's whole body the agent?

Not for this task. The agent selects delivery decisions; motors, wheels, sensors, and the physical world realize their consequences. A command is under the decision-maker's control, but its physical outcome is not arbitrarily selectable. Moving the interface to motor voltages would define a different, lower-level task.

In Python, objects are not a security boundary: code could technically overwrite environment attributes. The conceptual interface forbids that shortcut. The policy is allowed to select actions, not rewrite the transition table or reward values.

### Why must reward computation remain outside the agent?

Reward defines success for the task. If the agent could freely replace the delivery reward with an enormous constant, it would change the task instead of solving it. A reward routine can run on the same computer—or inside an organism's body—and still be conceptually external to its decision-making agent.

The agent may influence rewards **through allowed actions**. That is different from having arbitrary authority over the rule that computes them.

### Does “environment” mean “unknown”?

No. This lesson publishes the entire transition table. The agent–environment boundary concerns control, not knowledge. A known environment can still require difficult planning: knowing legal Rubik's cube moves does not supply a good sequence of moves. Learning is one way to improve decisions; a small known problem like this one can also be solved directly without learning.

### Can states contain memory, uncertainty, or mental information?

Yes. A state representation can include remembered observations, a task phase, or an estimate of where an object might be. A robot that loses sight of a customer could retain the last observed location rather than treating the current camera image as its entire basis for action.

However, a useful observation is not automatically a sufficient state for a Markov model. If relevant hidden history still changes the consequences of an action, the representation may need memory or a belief about hidden conditions. The toy delivery states avoid this issue by defining all transitions explicitly.

### Can actions be mental rather than physical?

Yes. Choosing to inspect a map, retrieve a memory, or allocate computation can be a decision to learn. Such an action changes information or internal processing rather than directly moving the robot. A model should specify its consequences and any time or reward cost; calling something a mental action does not make those costs disappear.

### Can the same robot contain several agents?

Yes. A route-selection agent might choose the distant customer, while a motor-control agent chooses wheel commands to follow that route. The route choice becomes part of the lower-level agent's input. Each task has its own states, actions, and rewards, so each has its own boundary. A high-level agent's action need not be a primitive physical action.

### Why does representation matter?

Suppose both `DEPOT` and `ROAD` were represented only as “not done.” The policy would lose the distinction between choosing a customer and completing an already chosen delivery; even their action sets differ. Conversely, irrelevant sensor detail can make learning unnecessarily difficult.

Good representations preserve distinctions that matter for decisions. Which distinctions matter depends on the task. Battery charge, deadlines, or customer identity are unnecessary in this toy model but could be essential in a realistic one.

### Are three signals enough for every decision-learning problem?

They provide a useful abstraction, not a guarantee of a useful formulation. Poor state information, inappropriate actions, or a reward that misses the intended goal can make the resulting task misleading or hard to learn. Saying “maximize reward” does not establish that the reward captures everything people actually care about.

For example, if delivery speed were rewarded without modeling safety, a high return would not prove safe behavior. The interface separates the signals clearly; choosing a faithful task remains a modeling responsibility.

## 4. Check your understanding

1. **After choosing to drive far at $t=0$, what is the next experience?** $R_1=-1$ and $S_1=\text{ROAD}$. The distant delivery reward arrives later, as $R_2=5$.
2. **Can the policy select “finish distant delivery” at the depot?** No. It is not in $\mathcal{A}(\text{DEPOT})$; the environment rejects it.
3. **If the robot knows the delivery reward is $5$, does that reward become part of the agent?** No. Knowledge of the rule does not confer control over it.
4. **What changes when we change `probability_far`?** The policy, not the environment or goal. To demonstrate learning, experience—not the reader alone—would need to drive the change.
5. **What would changing the reward from $5$ to $1$ mean?** A different task: the distant route would total $0$, so the nearby route would be better under the same episode-total objective.

The core distinction is: **states inform choices, actions express choices, and rewards evaluate consequences under a task the agent does not arbitrarily redefine.**
