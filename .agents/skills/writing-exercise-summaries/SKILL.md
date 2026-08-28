---
name: writing-exercise-summaries
description: Writing exercise lectures and summaries in markdown files. 
---

# Writing Exercise Summaries

## Purpose

Write concise instructional summaries for reinforcement learning exercises.

The skill is given an exercise number, reads the corresponding Python exercise file, determines the reinforcement learning concept being practiced, and writes or updates the exercise's Markdown documentation.

The output should read like a short university lecture on reinforcement learning: precise, mathematical when useful, intuitive, and focused on explaining *why* the observed behavior occurs rather than merely describing the code.

Do not claim to be Richard Sutton or directly impersonate him. Use a rigorous pedagogical style consistent with a graduate-level reinforcement learning course and with the conceptual clarity associated with Sutton and Barto's *Reinforcement Learning: An Introduction*.

---

## Input

The skill receives an exercise number, `n`.

Examples:

```text
4
exercise 4
Exercise 4
```

For integer exercise identifiers, zero-pad the Python filename to three digits.

Example:

```text
Exercise number: 4
Python source: /reinforcement_learning/exercises/exercise004.py
Markdown output: /reinforcement_learning/docs/exercises/exercise4.md
```

If an exercise number contains additional notation, such as a decimal exercise number, inspect the existing repository naming convention and follow it rather than inventing a new convention.

---

## Source File

Read:

```text
/reinforcement_learning/playground/exercise{number:0>3}.py
```

For example:

```text
/reinforcement_learning/playground/exercise004.py
```

The exercise source is the primary source of truth.

Read the entire exercise file before writing the summary.

If the exercise imports local functions, classes, constants, or experiment helpers whose behavior is necessary to understand the exercise, read those definitions as well.

Do not describe behavior that cannot be established from the code, comments, generated data, program output, or figures.

---

## Output File

Write the completed Markdown to:

```text
/reinforcement_learning/docs/exercises/exercise{n}.md
```

For example:

```text
/reinforcement_learning/docs/exercises/exercise4.md
```

Create the file if it does not exist.

If the file already exists, preserve useful existing content unless it conflicts with the actual implementation. Update the document rather than blindly replacing good material.

---

## Workflow

### 1. Read the Exercise

Read the complete exercise Python file.

Identify:

- the reinforcement learning problem being studied;
- the algorithm or update rule being used;
- important parameters;
- what variables are changed between experiments;
- what quantities are measured;
- what plots or figures are generated;
- the hypothesis or comparison the exercise is intended to demonstrate.

Relevant concepts may include:

- multi-armed bandits;
- action-value estimates;
- sample-average updates;
- constant step-size updates;
- epsilon-greedy action selection;
- optimistic initial values;
- exploration versus exploitation;
- nonstationary reward distributions;
- temporal-difference error;
- incremental estimation;
- bias and variance;
- convergence;
- recency weighting.

### 2. Determine the Teaching Objective

Determine the smallest set of concepts a student must understand to understand the exercise.

Do not turn a single exercise summary into a general reinforcement learning chapter.

The lecture should answer questions such as:

- What concept is this exercise testing?
- Why does the algorithm behave this way?
- What role does each important parameter play?
- What should the student expect before seeing the results?
- What subtle point is the exercise intended to reveal?

### 3. Inspect the Results

If the exercise generates figures, inspect both:

- the figure-producing code;
- the generated figure, when available.

Use the actual plotted behavior when explaining results.

Do not invent exact numerical values from a graph unless those values can be read reliably or are available directly from program output or experiment data.

When appropriate, run the exercise to reproduce its results.

Do not alter experiment parameters merely to create cleaner or more convenient results.

### 4. Write the Markdown

Use this general structure:

```markdown
# Exercise {n}

## Concept

<short lecture explaining the concept>

## Exercise Summary

<summary of what the exercise does and what is being compared>

![Average reward](figure.png)

**Results.** <explanation of what happened and why>
```

Do not force unnecessary sections.

A small exercise should remain a small document.

---

## Lecture Style

Write the `## Concept` section as a short lecture to a technically capable student studying reinforcement learning.

The lecture should:

- begin with the central idea;
- explain intuition before or alongside equations;
- introduce mathematical notation only when it clarifies the mechanism;
- connect equations to agent behavior;
- emphasize causal explanations;
- distinguish estimated values from true values;
- distinguish exploration behavior from learning behavior;
- point out bias, convergence, or transient behavior when relevant.

Prefer explanations such as:

> An optimistic initial value does not make the agent optimistic in a psychological sense. It changes the agent's action-value estimates so that untried actions remain more attractive than actions whose estimates have already been corrected by experience.

Avoid vague explanations such as:

> The agent explores because the values are high.

When an update equation is central to the exercise, show it in LaTeX.

For example:

```markdown
$$
Q_{n+1}(A_n)
=
Q_n(A_n)
+
\alpha
\left[
R_n-Q_n(A_n)
\right].
$$
```

Then explain the terms that matter to the exercise.

Keep the lecture concise. In most cases, approximately 2–5 paragraphs is sufficient.

---

## Exercise Summary

The `## Exercise Summary` section explains what the code actually does.

Describe:

- the environment or problem setup;
- the experimental conditions;
- important parameter values;
- what is held constant;
- what is varied;
- the number of runs or steps when relevant;
- what metric is plotted.

Do not narrate the Python line by line.

Prefer:

> The exercise compares greedy action selection under neutral and optimistic initial action-value estimates. Each agent interacts with the same class of 10-armed bandit problems, and average reward is measured over repeated runs.

Avoid:

> First the code creates an array. Then it enters a loop. Then it calls `select_action()`.

Mention function or variable names only when they help connect the explanation to the implementation.

---

## Graph Handling

Every graph included in the Markdown must be followed immediately by a results explanation.

For example:

```markdown
![Average reward](figure.png)

**Results.** The optimistic agent initially receives relatively poor rewards because all actions are estimated to be much better than they really are. Each disappointing reward lowers the estimate of the selected action, leaving untried actions comparatively attractive. This produces systematic exploration even though the policy itself is greedy.
```

Never leave a graph unexplained.

For each graph, explain:

1. what the axes represent;
2. which curves, bars, or markers correspond to which experimental conditions;
3. the important pattern;
4. the reinforcement learning mechanism responsible for that pattern;
5. any transient effects, convergence behavior, or surprising features.

Focus on explanation rather than visual narration.

Weak:

> The blue line goes up and then levels off.

Strong:

> Average reward rises after the initial exploration phase because the action-value estimates have become sufficiently accurate for the greedy policy to repeatedly select high-value actions.

---

## Explaining Spikes, Drops, and Transients

When a graph contains a spike, dip, oscillation, crossover, or other temporary effect, explicitly explain its cause.

Trace the effect through the algorithm:

```text
parameter choice
    ↓
action-value estimates
    ↓
action-selection behavior
    ↓
observed rewards
    ↓
update rule
    ↓
graph shape
```

For example, with optimistic initial values in a `k`-armed bandit:

- all actions begin with unrealistically high estimates;
- a greedy agent selects one of them;
- the observed reward is usually below the optimistic estimate;
- the selected action's estimate falls;
- untried actions remain artificially attractive;
- the agent therefore samples many or all actions;
- after those estimates are corrected, it can identify and exploit better actions;
- synchronized transitions across many independent runs can produce a visible spike in average reward near the point where initial exploration is exhausted.

Do not attribute a graph feature to randomness when there is a clear algorithmic explanation.

---

## Mathematical Accuracy

Use reinforcement learning terminology precisely.

Important distinctions include:

- $q_*(a)$: the true action value;
- $Q_t(a)$: the estimated action value at time $t$;
- $R_t$: the observed reward;
- $A_t$: the selected action;
- $\alpha$: the step size;
- $\epsilon$: the probability of exploratory action under an epsilon-greedy policy.

Do not refer to $Q_t(a)$ as the true reward.

Do not confuse an observed reward with an expected reward.

Do not describe a greedy policy as random exploration unless the mechanism actually produces that behavior.

---

## Results Writing

The results section should explain the experiment, not merely restate the graph.

A strong explanation should answer:

- What happened?
- Why did it happen?
- Which parameter or algorithmic mechanism caused it?
- Does the result agree with the expected reinforcement learning behavior?
- Is any unusual feature temporary or persistent?

When comparing algorithms, explain both their early behavior and asymptotic behavior when relevant.

For example:

- optimistic initial values may encourage strong early exploration;
- epsilon-greedy policies continue explicit exploration indefinitely;
- sample-average methods converge appropriately in stationary environments;
- constant step-size methods retain sensitivity to recent rewards and are more suitable for nonstationary environments.

---

## Tone

Write like an instructor explaining the exercise to an engineering or computer science student.

Be:

- concise;
- rigorous;
- direct;
- explanatory;
- technically precise.

Do not use motivational filler.

Do not praise the student.

Do not write conversational phrases such as:

- "As you can see..."
- "Pretty cool, right?"
- "Basically..."
- "Simply put..."

Do not write as though the reader already understands the result.

Teach the mechanism.

---

## Formatting

Use standard Markdown.

Use LaTeX for mathematical expressions.

Use backticks for:

- Python identifiers;
- file names;
- parameter names;
- literal values when appropriate.

Prefer descriptive graph alt text:

```markdown
![Average reward over time](figure.png)
```

rather than:

```markdown
![graph](figure.png)
```

However, preserve existing graph paths exactly unless the exercise itself has changed them.

Do not change image filenames merely for stylistic reasons.

---

## Verification

Before completing the task, verify that:

- the correct Python exercise file was read;
- the reinforcement learning concept matches the implementation;
- parameter values in the summary match the source code;
- every graph has a results explanation immediately below it;
- mathematical notation is correct;
- no results were invented;
- the Markdown output path is correct;
- the final document explains *why* the observed behavior occurs.

The final output of this skill is the updated Markdown exercise summary file.
