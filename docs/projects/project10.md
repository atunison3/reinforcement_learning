# Project 10 — Blackjack Card Counting

Source: `reinforcement_learning/projects/project010.py`.

This project implements full blackjack rounds against a dealer using a finite shoe, Hi-Lo counting, and a small tabular Monte Carlo learner. An episode includes **every hand created by splitting**. The agent can always select hit, stand, double, or split; inappropriate game actions receive a terminal reward of **−100**, rather than being hidden by an action mask.

This is separate from the earlier one-decision blackjack environments. A hit here can lead to another decision, and the shoe is not restored after each round. Before any player decision, the dealer peeks for blackjack when showing an ace or ten. A revealed dealer natural immediately terminates and **discards the episode from learning**.

## Run

From the repository root, with the package and its dependencies installed:

```bash
python -m reinforcement_learning.projects.project010 --episodes 10000 --seed 42
```

The script trains an epsilon-greedy agent with `epsilon=0.1`. `--episodes` counts dealt rounds; the output reports retained training rounds and discarded dealer-blackjack rounds separately. Mean training return excludes discarded deals rather than treating them as zero-return samples. It also prints invalid-action rounds, observed states, shuffle count, and the final running count. It does not save a model or plots. These statistics exclude dealer-natural outcomes and include exploration and artificial penalties; they are **not an estimate of casino profitability**.

## State representation

Each observation is a frozen, hashable `BlackjackState`:

```python
BlackjackState(
    hand=HandCategory.NINE,
    can_double=True,
    can_split=False,
    dealer_upcard=10,
    count=CountBucket.POSITIVE,
)
```

| Field | Meaning |
| --- | --- |
| `hand` | Category of the currently active player hand, as defined below |
| `can_double` | Two cards and not an original natural blackjack |
| `can_split` | Two equal-valued cards, with room below the configured total-hand limit |
| `dealer_upcard` | Visible dealer card: ace = 1, cards 2–9, ten-valued cards = 10 |
| `count` | Running count `>0`, or `<1` (zero and negative integers) |

The flags describe the rules but do **not** restrict the agent's action selection. `environment.actions` always contains all four actions.

### Hand categories and precedence

Cards use values 1–10, with jacks, queens, and kings all represented as 10. Aces count as 11 when that avoids a bust, otherwise as 1. Card order does not matter.

| Category group | Labels | Classification |
| --- | --- | --- |
| Natural | `blackjack` | Original, unsplit two-card ace + ten; takes precedence over all totals |
| Pairs | `ace ace`, `two two`, `three three`, `four four`, `five five`, `six six`, `seven seven`, `eight eight`, `nine nine`, `ten ten` | Exact two-card pairs; classified separately even when a split limit prevents splitting |
| Composition-specific eights | `six two`, `five three` | Exact two-card compositions; `four four` is already in the pair group |
| Other eight | `8 (other composition)` | Covers multi-card eights such as 2 + 2 + 4 |
| Soft totals | `ace two`, `ace three`, `ace four`, `ace five`, `ace six`, `ace seven`, `ace eight` | Usable-ace totals 13–19, after checking pairs and natural blackjack |
| Ordinary totals | `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16` | Remaining hands grouped by total, not composition |
| Catch-all totals | `<8`, `>16` | Remaining non-busted hands below 8 or above 16 |

Examples:

- 5 + 4 and 3 + 6 are both `9`.
- 6 + 2, 5 + 3, and 4 + 4 remain distinct.
- 5 + 5 is `five five`, not the ordinary `10` bucket.
- Ace + 2 + 4 has soft total 17 and uses `ace six`, but `can_double=False` and `can_split=False`.
- Ace + 9 and hard 17–21 fall into `>16` unless an earlier category applies.
- Ace + ten after splitting is ordinary 21 in `>16`, not natural blackjack.
- Busted hands never become decision states; play advances to another split hand or ends the round.

The extra `blackjack` category is necessary for learning the special stand-only rule. The extra ordinary-eight category makes the representation cover multi-card hands as well as the requested two-card eights.

## Game rules

Default configuration:

```python
from reinforcement_learning.projects.project010 import BlackjackEnvironment

environment = BlackjackEnvironment(
    decks=6,
    max_hands=4,
    penetration=0.75,
    seed=42,
)
```

- **Dealer:** stands on all 17s, including soft 17. One dealer hand settles all player hands.
- **Hit:** draw one card; continue choosing actions unless busted. A non-natural 21 may still hit, although this is usually undesirable.
- **Stand:** finish the active hand.
- **Double:** allowed on any two-card non-natural hand, including after a split. Double the stake, draw exactly one card, then finish that hand.
- **Split:** two equal card values become two hands. Both receive one visible new card immediately; play the first child before the second. Ten-valued face cards can split because ranks are deliberately collapsed.
- **Re-split:** allowed up to four total hands by default. `max_hands` supports limits from one to four, including split descendants.
- **Split aces:** may hit, double, or re-split under the same rules as other split hands. They are not restricted to a single draw.
- **Naturals:** only an original, unsplit two-card 21 qualifies for 3:2. If the dealer does not have blackjack, the agent must choose stand to receive this payoff.
- **Dealer peek:** with an ace or ten upcard, the dealer checks the initial hole card **before the player acts**. A natural is revealed immediately, the round terminates, and the episode is discarded—even if the player also has blackjack. No player hit, stand, double, or split occurs on that deal. A negative peek leaves the hole card hidden until settlement.
- No insurance, surrender, bankroll, wager-sizing action, or other players are modeled. Each initial or newly split hand starts with a one-unit stake.

### Rewards

All amounts are **net payoffs**, not gross returned stakes:

| Outcome | Reward for that hand |
| --- | ---: |
| Ordinary win / loss / push | +1 / −1 / 0 |
| Doubled win / loss / push | +2 / −2 / 0 |
| Standing original blackjack against a non-natural dealer | +1.5 |
| Dealer blackjack revealed at the initial peek, including mutual naturals | Episode discarded; no learning reward |
| Any action other than stand on original blackjack | −100 for the entire round |
| Split on a non-pair, split beyond the hand limit, or double with three or more cards | −100 for the entire round |

Player busts lose their stake even if the dealer would bust. Dealer naturals never reach player decisions or settlement in this project. A dealer who draws to 21 later is not a natural; ordinary comparisons, including pushes against player 21, still apply. The dealer need not draw when all player hands are busted or the only player hand is a standing natural.

Discarding a dealer natural is a **training filter**, not a simulated cash settlement: neither a normal loss nor a mutual-natural push becomes a learning sample. It is not the −100 penalty, because the player made no invalid decision. The four dealt cards remain removed from the shoe, and the revealed hole card is counted once.

An invalid game action **terminates the whole round**, including any pending split hands. Its final reward is exactly −100, not −100 plus other hand results. Already exposed cards remain counted and the dealer hole card is revealed once; no additional dealer play occurs on that aborted round. Unknown action identifiers are API errors rather than game actions.

## Full-round episodes and split credit

Valid nonterminal decisions return zero reward. Only after all player hands finish does the environment settle the round and return the sum of their payoffs:

```text
split 8,8 → 8,10 and 8,10      reward 0
stand on first 18             reward 0
stand on second 18
  dealer has 17               reward +2; episode ends
```

The dealer is not re-dealt between split hands. A win and a loss sum to 0; two ordinary wins sum to +2; two doubled wins sum to +4.

For an undiscounted return,

$$
G_t = \sum_{k=t}^{T-1} R_{k+1}.
$$

In the example, the return associated with the split decision is $0+0+2=2$. Ending an episode after the first child hand would incorrectly omit the other child's outcome.

`MonteCarloAgent.learn` processes the full trajectory backward with $\gamma=1$ and averages observed returns for each state–action pair. It does not manually add another split bonus: that would count the same outcome twice.

### Environment API

```python
from reinforcement_learning.projects.project010 import Action, BlackjackEnvironment

environment = BlackjackEnvironment(seed=42)
state = environment.reset()
if state is None:
    print("Episode discarded:", environment.discard_reason)
else:
    result = environment.step(Action.STAND)
    print(result.reward, result.terminated)
    print(result.hand_rewards)  # Terminal per-hand outcomes for diagnostics.
```

`reset()` returns a `BlackjackState` only when the player has a decision to make. On dealer blackjack it returns `None`, sets `environment.discard_reason` to `"dealer_blackjack"`, and ends the round immediately. Calling `step()` on that discarded deal raises `RuntimeError`. The next `reset()` deals the next round and clears the reason if it is playable; it does not silently retry or restore the discarded cards.

`step` returns a `StepResult` with:

- `state`: the next active-hand observation, or `None` on termination;
- `reward`: zero until termination, then total net round payoff or −100;
- `terminated`: whether the entire round has finished;
- `hand_rewards`: settled per-hand payoffs for valid terminal rounds, otherwise empty;
- `invalid_reason`: explanation of a penalized decision, otherwise `None`.

Call `reset()` between rounds, not between split hands. Resetting an active round or acting after termination raises `RuntimeError`.

For deterministic experiments, `reset(shoe=...)` accepts a **complete fresh shoe** in draw order with the exact configured card multiplicities. The deal order is player card, player card, dealer upcard, dealer hidden card. Installing a fresh shoe resets the count. Ordinary resets preserve the existing shoe unless a reshuffle is due.

### Training from the whole trajectory

```python
from reinforcement_learning.projects.project010 import (
    BlackjackEnvironment,
    MonteCarloAgent,
    run_episode,
)

environment = BlackjackEnvironment(seed=42)
agent = MonteCarloAgent(epsilon=0.1, seed=43)

for _ in range(10000):
    episode = run_episode(environment, agent)  # Updates only playable rounds.
    if not episode:
        continue  # Dealer blackjack: no decisions or learning updates.

# agent.q[state][action] is an average full-round return, not a payout table.
```

`run_episode` returns an empty tuple for a discarded dealer-natural deal, without calling either `choose_action` or `learn`. It produces no transition, fake action, or zero-reward sample. Other episodes still update the learner, including −100 penalties for actual invalid choices.

On playable rounds, the agent samples all four actions, including those inconsistent with the state flags, so penalties become learning experience. With nonzero epsilon, it will continue making some invalid choices even after their negative values have been learned.

## Card counting and the shoe

The environment maintains the full integer **Hi-Lo running count** internally:

| Exposed card | Count change |
| --- | ---: |
| 2–6 | +1 |
| 7–9 | 0 |
| Ten-valued card or ace | −1 |

Only its sign bucket is passed to the agent: `POSITIVE` for `>0`, `NON_POSITIVE` for `<1`. The full count remains available through `environment.running_count` for diagnostics. This is a running count, **not a true count** divided by decks remaining.

Count updates include initial visible player cards, the dealer upcard, player hit/double cards, both newly dealt split cards, revealed dealer hole cards, and dealer draws. A positive dealer-blackjack peek reveals and counts the hole card immediately, including on discarded rounds. A negative peek does not expose or count the hole card. Cards already exposed are not counted again when a hand is split or revisited. The hidden dealer card and undrawn cards do not affect the observation before they are exposed.

The finite shoe is sampled **without replacement** and persists across episodes. The count resets only with a fresh shoe. The default cut point is 75% dealt; a conservative reserve can trigger an earlier reshuffle to guarantee enough cards for all allowed split hands and the dealer. Shuffling occurs only before a round, never midway through one. `cards_remaining` and `shuffle_count` are available for diagnostics.

## Deliberate limitations

This observation is a compact learning representation, **not a complete Markov state**:

- The `>16` bucket merges hard 17 through 21 and some soft totals with different strategic values.
- A count sign does not determine the remaining shoe composition.
- Pending split-hand identities, already completed outcomes, and the exact number of hands are not included, although `can_split` indicates when the cap has been reached.
- Every decision receives the final whole-round return, including outcomes of other hands; the state does not describe all those sources of reward.

These choices preserve the requested simple state rather than implying that a tabular learner can recover an optimal blackjack strategy from it. The learner also optimizes individual round returns, not consequences for future rounds in the same shoe. Count-based bet sizing is outside this version.

## Tests

```bash
python -m unittest discover -s tests -p "test_project010.py" -v
```

Tests cover category precedence, multi-step hits, legal and invalid actions, dealer peeks with ace and ten upcards, discarded mutual naturals, absence of actions/updates on discarded deals, training-statistic filtering, split/re-split handling, double-after-split payoffs, combined return credited to splitting, finite-shoe continuity, hidden-card counting, reshuffles, and reproducible training.
