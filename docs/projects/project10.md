# Project 10 — Blackjack Card Counting

Source: `reinforcement_learning/projects/project010.py`.

This project learns blackjack decisions with **masked, one-step Double Q-learning**, a finite shoe, and Hi-Lo counting. One episode is a complete dealt round, including every split descendant—not an entire shoe. The shoe and running count persist between rounds.

The learner distinguishes **exact hand totals and usable aces**, while preserving pair and composition-specific categories. Hard 17, 18, 19, 20, and 21 no longer share a Q-value. Completed and pending split hands are part of the observation, so acting on the first split hand differs from acting on the last.

## Run

From the repository root, with the package installed:

```bash
python -m reinforcement_learning.projects.project010 --episodes 100000 --epsilon 0.1 --alpha 0.1
```

The same command works in macOS/Linux shells and Windows PowerShell. The existing epsilon default is **0.0**, preserved from the prior configuration. Pass `--epsilon 0.1` to explore legal moves on 10% of decisions. The default step size is `--alpha 0.1`; gamma is fixed at 1. Both Q estimators start at 10 for untried entries.

The CLI prints final strategy tables and writes:

- [Strategy Markdown](assets/project_10_strategy.md)
- [Strategy CSV](assets/project_10_strategy.csv)

The default directory is **`docs/projects/assets/`**. Existing files are replaced at checkpoints and completion. Back up prior results, or use `--output-dir results/blackjack-double-q` to keep an experiment separate. Existing checked-in reports may describe an older algorithm until a new run replaces them.

For the requested billion-round run, start a **fresh strategy** by omitting `--load-strategy`:

```bash
python -m reinforcement_learning.projects.project010 --episodes 1000000000 --epsilon 0.1 --alpha 0.1 --output-dir docs/projects/assets
```

This is a long-running job. State counts, memory use, CSV size, and checkpoint-writing time can grow substantially as more split contexts are encountered. No billion-round run is started automatically by implementing this project.

## Why Double Q-learning replaces Monte Carlo

The previous learner averaged the **actual full-round return** into every visited state–action value. A later exploratory mistake therefore affected earlier actions, even if those earlier actions were good choices.

The new learner separates two policies:

- **Behavior:** epsilon-greedy exploration among legal actions, using the mean of `Q_A` and `Q_B` for exploitation.
- **Target:** a greedy legal continuation, independent of which exploratory action is selected next.

On each transition, the agent randomly selects estimator A or B to update. When updating A:

$$
a^* = \arg\max_{b\in A(s')} Q_A(s',b)
$$

$$
Q_A(s,a) \leftarrow Q_A(s,a) + \alpha\left[r + Q_B(s',a^*) - Q_A(s,a)\right].
$$

The B update exchanges A and B. Greedy ties are broken randomly. At termination, the target is **only the observed reward**, with no bootstrap term. This separates action selection from evaluation to reduce maximization bias; it does not eliminate estimation error.

### Example: hit 7, then explore stand on 10

1. Hit on 7, receive a 3, and observe a three-card 10.
2. Immediately update hit-on-7 using the best estimated legal continuation at 10, selected by one estimator and evaluated by the other.
3. If the behavior policy then explores stand and loses, update stand-on-10 toward −1.
4. Do **not** replay that loss backward as the return for hit-on-7.

The update happens before selecting the next behavior action. The learner still accounts for genuine card risk and losses caused directly by an action. It learns expected returns—not the best lucky outcome—and does not discard losing samples merely because an action was greedy.

`alpha` is a fixed step size, not `1 / visits`. Update counts are diagnostics and resume data, not Monte Carlo sample-average weights. A first update does not necessarily replace the initial 10. With fixed alpha, noise persists; there is no claim of exact convergence.

## Action masking and exploration

The shared `allowed_actions(state)` helper is used by behavior, both Double Q target selections, strategy reports, and environment validation:

| Situation | Legal actions |
| --- | --- |
| Original natural blackjack | Stand only |
| Non-natural hand | Hit and stand |
| `can_double=True` | Also double |
| `can_split=True` | Also split |

With epsilon $\epsilon$, $m$ legal moves, and a unique greedy move, that move is chosen with probability $1-\epsilon+\epsilon/m$; each other legal move has probability $\epsilon/m$. Illegal moves have probability zero even if their stored values are larger.

Masking teaches **legality, not strategy**. Hitting a hard 20, doubling a split 21, or splitting tens remains available when legal. Epsilon zero disables random exploration, but optimistic initialization and greedy ties can still cause different legal actions to be tried. Neither epsilon zero nor optimism guarantees sufficient coverage.

Illegal direct calls to `step()` raise `ValueError` before changing cards, counts, or round status. They generate neither a reward nor a transition. There are no artificial invalid-action penalties or retries during normal training. An invalid custom-policy call propagates the error; any earlier, valid online updates remain, but no fabricated loss is assigned to them.

## Exact decision state

`BlackjackState` is immutable and hashable:

| Field | Meaning |
| --- | --- |
| `hand` | Composition/category label, retained for pairs and requested special hands |
| `total` | Exact current total, with aces valued to avoid busting |
| `usable_ace` | Whether one ace currently counts as 11 |
| `can_double` | Two cards and not an original natural |
| `can_split` | A two-card pair with room below the total-hand limit |
| `dealer_upcard` | Visible dealer card, ace = 1 and ten-valued cards = 10 |
| `count` | Hi-Lo running-count bucket `>0` or `<1` |
| `max_hands` | Configured total-hand limit, including descendants |
| `completed_hands` | Ordered tuple of `(final_total, stake)` for finished split hands |
| `pending_hands` | Ordered tuple of the two exposed cards in each pending split hand |

Within each pending hand, card values are sorted because card order does not affect play. Pending hands always have two cards and a unit stake. Completed hands may be busted or doubled; total and stake suffice for their eventual settlement. All hands in a split round are ordinary hands, so a completed split 21 never receives a natural's 3:2 payout.

The active hand always has a unit stake at a decision: doubling immediately finishes that hand. Its eligibility flags capture the relevant two-card versus multi-card distinction. The sizes of the completed and pending tuples identify the position within the split round and total number of hands.

### Exact totals and retained labels

- `six two`, `five three`, and `four four` remain separate hard-eight compositions.
- Every pair retains its rank-specific category.
- Original natural blackjack remains separate from ordinary 21.
- Usable-ace hands remain distinguishable from hard hands of the same total.
- The internal `<8` and `>16` category labels remain for compatibility with composition classification, **but no longer aggregate Q-values**: `total` and `usable_ace` are independent parts of the key.
- Busts are not decision states; play moves to the next hand or settlement.

For example, hard 20 and soft 20 can share the category `>16`, but have different `usable_ace` values and therefore independent estimates. Hard 17 and hard 20 have different `total` values. Two successive split 21s have different completed/pending contexts.

No hidden dealer card or undrawn shoe order is included in the state.

## Game rules and rewards

Defaults: six decks, dealer stands on soft 17, up to four hands, 75% penetration. Doubling after splitting, re-splitting, and hitting split aces are allowed. Ten-valued ranks are collapsed to 10.

- **Hit:** draw one card and continue unless busted. Ordinary 21 can still hit.
- **Stand:** finish the active hand.
- **Double:** double the stake, draw exactly one card, and finish the active hand.
- **Split:** replace a pair with two hands; both receive a visible second card immediately. Play the first child before the second.
- **Re-split:** allowed until the total-hand cap; completed hands still count toward the cap.
- **Original natural:** stand to receive +1.5 if the dealer does not have blackjack.
- **Dealer peek:** with an ace or ten showing, inspect the hole before any player action. A dealer natural, including mutual naturals, ends and **discards the deal from learning**. A negative peek keeps the hole hidden until settlement.
- No insurance, surrender, bankroll, variable initial wagers, or other players are modeled.

All rewards are **net payoffs**:

| Outcome | Per-hand payoff |
| --- | ---: |
| Ordinary win / loss / push | +1 / −1 / 0 |
| Doubled win / loss / push | +2 / −2 / 0 |
| Standing original natural | +1.5 |
| Dealer natural at the initial peek | Discarded; no action or learning reward |

Player busts lose their stakes regardless of the dealer's outcome. Split ace-ten is ordinary 21. A dealer-drawn 21 is not a natural and uses ordinary comparisons. Dealer draws are unnecessary when every player hand has busted or the only hand is a standing natural.

Discarding dealer naturals is a **training filter**, not casino cash settlement. Such deals consume and count their exposed cards but do not become zero-reward learning samples. Consequently, training statistics are not an estimate of casino profitability.

### Full-round reward and split credit

Valid intermediate transitions return zero. Final settlement returns the sum of all hand outcomes against the same dealer:

```text
split 8,8 into 8,10 and 8,10   reward 0; next state has one pending hand
stand on first 18              reward 0; next state includes completed (18, 1)
stand on second 18
  dealer has 17                reward +2; round terminates
```

Double Q-learning propagates the expected aggregate payoff through these context-specific states over repeated updates. It does **not** immediately assign +2 to every preceding action after one episode. There is no extra split bonus and no duplicate payout. Two doubled wins total +4; a win and a loss total zero.

## Python API

```python
from reinforcement_learning.projects.project010 import (
    BlackjackEnvironment,
    DoubleQLearningAgent,
    run_episode,
)

environment = BlackjackEnvironment(seed=42)
agent = DoubleQLearningAgent(seed=43, epsilon=0.1, alpha=0.1)
for _ in range(10000):
    transitions = run_episode(environment, agent)
    # () means a discarded dealer-natural deal; otherwise each transition
    # has already produced exactly one online Double Q update.
```

- `reset()` returns `BlackjackState`, or `None` with `discard_reason="dealer_blackjack"`.
- `step(action)` returns `StepResult(state, reward, terminated, hand_rewards)`.
- `Experience(state, action, reward, next_state)` records one transition. `next_state=None` means terminal.
- `agent.learn(experience)` performs one update, not an episode-return sweep.
- `agent.action_values(state)` returns `(Q_A + Q_B) / 2` as an action-indexed list.
- `agent.action_visits(state)` returns the sum of the two estimators' update counts.
- `q_a`, `q_b`, `visits_a`, and `visits_b` hold the estimator-specific data.

Call `reset()` between rounds, not between split hands. Resetting an active round or stepping an inactive round raises `RuntimeError`. For deterministic tests, `reset(shoe=...)` accepts a complete correctly composed fresh shoe in draw order: player, player, upcard, hole, subsequent draws.

## Reading the expanded strategy tables

The Markdown contains matrices by dealer upcard (2–10, A), count bucket, double/split eligibility, and hand limit. Rows use **exact hard/soft totals**, including 17, 18, 19, 20, and 21. Pair and special-composition labels appear alongside the total. Structurally possible but unseen rows remain visible.

Only possible combinations are shown: for example, hard 21 cannot be an original two-card hand, and hard 4 is a pair of twos. Two- and multi-card soft totals have separate eligibility tables.

**The matrices describe one unsplit hand with no completed or pending hands.** They must not be reused for split children. The Markdown reports the number of observed split-context states; the CSV stores each full context and its own recommendation rather than silently averaging contexts together. Use `strategy_cell(agent, state)` to query a live split-hand state.

| Symbol | Meaning |
| --- | --- |
| `H`, `S`, `D`, `P` | Hit, stand, double, split |
| `?` | No sampled legal action, or a completely untried legal action leads/ties |
| `*` | At least one legal action lacks an update in one of the two estimators |
| `/` | Greedy legal ties, such as `H/S` |

Recommendations use the mean of both estimators without random exploration. Illegal actions never win a recommendation or count against coverage. Initial values are optimistic estimates, not observations. Coverage is not confidence or proof of convergence.

## Versioned CSV and resume

**Old Monte Carlo CSVs are incompatible.** They lack exact totals, split context, and separate estimators. Expanding a coarse row into multiple exact states or duplicating its mean into both estimators would invent information. The loader rejects those files before training or output, with a message to start fresh. Existing files are not automatically converted or deleted.

New CSVs include:

- `format_version=2` and `algorithm=double_q`;
- every state field, with completed/pending tuples encoded as JSON arrays;
- the derived recommendation and total action-update count;
- for each action, combined `q_...`, `q_a_...`, `visits_a_...`, `q_b_...`, and `visits_b_...`.

Only **observed exact states** are stored; this is no longer the old fixed 1,160-row coarse grid. Untried estimator cells are blank with zero updates and reload as 10. Sampled zeros remain zeros. Both estimator tables and both update-count tables are required for continued learning.

The loader validates the complete file before replacing agent data, including version, algorithm, state structure, contexts, duplicate states, finite values, legal-action update counts, and consistency of derived values/counts. Recommendations are regenerated, not used as training data. An empty version-2 checkpoint has a header and no state rows.

To resume a **new-format** CSV:

```bash
python -m reinforcement_learning.projects.project010 --load-strategy docs/projects/assets/project_10_strategy.csv --episodes 100000 --epsilon 0.1 --alpha 0.1
```

`--episodes` means additional dealt rounds. Estimators and counts resume; the shoe, count, RNG streams, and per-run statistics start fresh. Epsilon and alpha come from the current CLI arguments, not the CSV. Loading does not restore an exact trajectory or automatically repeat the prior policy settings.

## Checkpoints and runtime reporting

- Progress prints after each **10,000 dealt rounds**, plus a final partial block.
- Both Markdown and CSV are saved after each **100,000 completed dealt rounds** and at completion.
- Discarded deals count toward both schedules. Saves occur between rounds, after all player decisions and their updates.
- The same files in `--output-dir` are overwritten. A run ending exactly on a checkpoint boundary writes once, not twice.
- A 250,000-round run saves at 100,000, 200,000, and 250,000. Resumed schedules start at zero for the new invocation.
- Each checkpoint summary describes rounds completed so far, while estimator counts include loaded history.
- Only final matrices are printed; intermediate checkpoints print a short saved-path confirmation.
- Training time includes progress and checkpoint overhead, but excludes loading and final export. A checkpoint's timestamp is taken before its own write; later timings include that cost.
- The normalized per-1,000 metric is elapsed training seconds × 1,000 / dealt rounds. Total program time includes loading, training, and final reports, but excludes imports and its own final print.

A completed CSV checkpoint can resume learning after interruption. Changes after the most recent completed save are lost; checkpoints are not full environment/RNG snapshots.

## Counting and remaining limitations

Hi-Lo counts exposed 2–6 as +1, 7–9 as zero, and aces/tens as −1. Both split draws are visible immediately. The hole card is counted only when revealed. A positive dealer peek reveals it before discarding; a negative peek does not reveal it. Cards are never counted twice because a hand was split.

The shoe is finite and sampled without replacement. A reshuffle occurs only between rounds, at penetration or earlier for a conservative full-round card reserve. The full integer running count is available for diagnostics; only its sign is in the learned observation. This is not a true count per remaining deck.

The richer state fixes hard/soft-total and split-context aliasing, but it is **still not a complete Markov description of a finite shoe**. It omits detailed remaining-card composition and the hidden dealer-card belief. Double Q-learning, finite data, fixed alpha, and optimistic initialization do not guarantee optimal blackjack play. The model optimizes a round, not effects on future rounds or count-based wager sizing.

Evaluate a frozen agent with epsilon zero separately from exploratory training. A bad current greedy estimate is still an estimate—not evidence that the underlying action is truly best.

## Tests

```bash
python -m unittest discover -s tests -p "test_project010*.py" -v
```

Tests cover blackjack rules, naturals and peeks, masks, finite-shoe counting, exact totals and soft/hard separation, split contexts and doubled stakes, both Double Q update directions, cross-estimator targets, terminal handling, online update order, protection from later exploratory losses, delayed split-credit propagation, reproducibility, exact report rows, versioned resume, atomic validation, legacy rejection, checkpoints, and timing.
