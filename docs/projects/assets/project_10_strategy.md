# Project 10 — Learned Action Tables

- Loaded strategy: docs/projects/assets/project_10_strategy.csv; states: 968; action visits: 42639383
- Dealt rounds: 5000000; training rounds: 4762624
- Policy: epsilon-greedy; epsilon: 0.1; initial action value: 10
- Training time this run: 210.144 s
- Average time per 1,000 dealt episodes (normalized): 0.042 s
- Seed: 42; decks: 6; max hands: 4
- Discarded dealer-blackjack rounds: 237376
- Mean training return (including penalties, excluding discarded rounds): -2.525
- Invalid-action rounds: 117440; observed states: 969
- Shoe shuffles: 106885; running count: 12
- Exploratory training statistics are not an estimate of casino profitability.

Training policy: epsilon-greedy (epsilon=0.1); untried Q = 10.
Tables show greedy learned choices, without random exploration; these are not proven optimal play.

`H` = hit; `S` = stand; `D` = double; `P` = split; `A` = dealer ace.
`?` = unlearned: unseen state or an untried action ties/leads using its initial 10.
`*` = not all four actions have been sampled; `/` = tied best choices; `!` = choice violates game rules.
An unmarked cell is not a confidence guarantee; inspect the CSV visit counts and Q estimates.
Impossible hand/flag combinations are omitted. All four actions remain unmasked during learning.

## Count <1; double no; split no

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <8 | S | S | S | S | S | S | S | S | S | H |
| 8 (other composition) | S | S | S | S | S | S | S | S | S | S |
| 9 | S | S | S | S | S | S | S | S | S | S |
| 10 | S | S | S | S | S | S | S | S | S | S |
| 11 | S | S | S | S | S | S | S | S | S | S |
| 12 | S | S | S | S | S | S | S | S | S | S |
| 13 | S | S | S | S | S | S | S | S | S | S |
| 14 | S | S | S | S | S | S | S | S | S | S |
| 15 | S | S | S | S | S | S | S | S | S | S |
| 16 | S | S | S | S | S | S | S | S | S | S |
| >16 | S | S | S | S | S | S | S | S | S | S |
| ace two | S | S | S | S | S | S | S | S | H | H |
| ace three | S | S | S | S | S | S | S | S | S | S |
| ace four | S | S | S | S | S | S | S | S | S | S |
| ace five | S | S | S | S | S | S | S | S | S | S |
| ace six | S | S | S | S | S | S | S | S | S | S |
| ace seven | S | S | S | S | S | S | S | S | S | S |
| ace eight | S | S | S | S | S | S | S | S | S | S |
| blackjack | S | S | S | S | S | S | S | S | S | S |

## Count <1; double yes; split no

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <8 | S | S | S | S | S | S | S | S | S | S |
| six two | D | D | D | D | D | D | S | S | S | S |
| five three | D | S | D | D | D | D | D | S | S | S |
| four four | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 9 | D | D | D | D | D | D | D | D | D | D |
| 10 | D | D | D | D | D | D | D | D | D | D |
| 11 | D | D | D | D | D | D | D | D | D | D |
| 12 | S | S | S | S | S | S | S | S | S | S |
| 13 | S | S | S | S | S | S | S | S | S | S |
| 14 | S | S | S | S | S | S | S | S | S | S |
| 15 | S | S | S | S | S | S | S | S | S | S |
| 16 | S | S | S | S | S | S | S | S | S | S |
| >16 | S | S | S | S | S | S | S | S | S | S |
| ace ace | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| two two | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| three three | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| five five | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| six six | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| seven seven | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| eight eight | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| nine nine | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| ten ten | ? | ? | ? | ? | ? | ? | ? | ? | S | ? |
| ace two | D | D | D | D | D | D | D | S | D | S |
| ace three | D | D | D | D | D | D | D | D | S | S |
| ace four | D | D | D | D | D | D | D | S | S | D |
| ace five | D | S | D | D | D | D | D | D | D | S |
| ace six | D | D | D | D | D | D | D | D | S | S |
| ace seven | S | S | S | D | D | S | S | S | S | S |
| ace eight | S | S | S | S | S | S | S | S | S | S |

## Count <1; double yes; split yes

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four four | D | D | D | D | D | D | D | S | S | S |
| ace ace | D | D | D | D | D | D | D | S | D | S |
| two two | S | S | S | S | S | S | S | S | S | S |
| three three | S | S | S | S | S | S | S | S | S | S |
| five five | D | D | D | D | D | D | D | D | D | D |
| six six | S | S | S | S | S | S | S | S | S | S |
| seven seven | S | S | S | S | S | S | S | S | S | S |
| eight eight | S | S | S | S | S | S | D | S | S | S |
| nine nine | S | S | S | S | S | S | S | S | S | S |
| ten ten | S | S | S | S | S | S | S | S | S | S |

## Count >0; double no; split no

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <8 | S | S | S | S | H | S | H | S | S | S |
| 8 (other composition) | S | S | S | S | S | S | S | S | S | S |
| 9 | S | S | S | S | S | S | S | S | S | S |
| 10 | S | S | S | S | S | S | S | S | S | S |
| 11 | S | S | S | S | S | S | S | S | S | S |
| 12 | S | S | S | S | S | S | S | S | S | S |
| 13 | S | S | S | S | S | S | S | S | S | S |
| 14 | S | S | S | S | S | S | S | S | S | S |
| 15 | S | S | S | S | S | S | S | S | S | S |
| 16 | S | S | S | S | S | S | S | S | S | S |
| >16 | S | S | S | S | S | S | S | S | S | S |
| ace two | S | H | S | H | H | H | S | H | S | S |
| ace three | S | S | S | S | S | H | S | S | S | H |
| ace four | S | S | H | S | S | S | S | S | S | S |
| ace five | S | S | S | S | S | S | S | S | S | S |
| ace six | S | S | S | S | S | S | S | S | S | S |
| ace seven | S | S | S | S | S | S | S | S | S | S |
| ace eight | S | S | S | S | S | S | S | S | S | S |
| blackjack | S | S | S | S | S | S | S | S | S | S |

## Count >0; double yes; split no

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <8 | S | S | S | S | D | S | S | S | S | S |
| six two | D | D | D | D | D | D | D | S | S | S |
| five three | D | D | D | D | D | D | D | S | S | S |
| four four | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 9 | D | D | D | D | D | D | D | D | D | D |
| 10 | D | D | D | D | D | D | D | D | D | D |
| 11 | D | D | D | D | D | D | D | D | D | D |
| 12 | S | S | S | S | S | S | S | S | S | S |
| 13 | S | S | S | S | S | S | S | S | S | S |
| 14 | S | S | S | S | S | S | S | S | S | S |
| 15 | S | S | S | S | S | S | S | S | S | S |
| 16 | S | S | S | S | S | S | S | S | S | S |
| >16 | S | S | S | S | S | S | S | S | S | S |
| ace ace | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| two two | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| three three | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| five five | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| six six | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| seven seven | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| eight eight | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| nine nine | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| ten ten | ? | ? | ? | D | ? | ? | ? | ? | ? | ? |
| ace two | D | D | D | D | D | D | D | D | D | D |
| ace three | D | D | D | D | D | D | D | D | D | S |
| ace four | D | D | D | D | D | D | D | D | S | S |
| ace five | D | D | D | D | D | D | D | D | D | S |
| ace six | D | D | D | D | D | D | D | S | S | S |
| ace seven | D | D | D | D | D | S | S | S | S | S |
| ace eight | S | D | S | S | D | S | S | S | S | S |

## Count >0; double yes; split yes

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four four | S | D | S | D | D | D | S | S | S | S |
| ace ace | S | D | D | D | D | D | D | S | S | D |
| two two | S | S | S | S | S | S | S | S | S | S |
| three three | S | S | S | S | S | S | S | S | S | S |
| five five | D | D | D | D | D | D | D | D | D | D |
| six six | S | S | S | S | S | D | D | S | S | S |
| seven seven | S | S | S | S | S | S | S | S | S | S |
| eight eight | S | S | S | S | S | S | S | S | S | S |
| nine nine | S | S | S | S | S | S | S | S | S | S |
| ten ten | S | S | S | S | S | S | S | S | S | S |
