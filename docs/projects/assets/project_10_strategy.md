# Project 10 — Double Q-Learning Strategy

- Loaded strategy: docs/projects/assets/project_10_strategy.csv; states: 809600; action visits: 18161340
- Dealt rounds: 6000000; training rounds: 5715452
- Algorithm: Double Q-learning; epsilon: 0.1; alpha: 0.1; gamma: 1; initial Q: 10
- Training time this run: 812.189 s
- Average time per 1,000 dealt episodes (normalized): 0.135 s
- Seed: 42; decks: 6; max hands: 4
- Discarded dealer-blackjack rounds: 284548
- Mean training return (excluding discarded rounds): -0.247
- Observed states: 1164626
- Shoe shuffles: 151821; running count: 3
- Exploratory training statistics are not an estimate of casino profitability.

Policy: masked epsilon-greedy; epsilon: 0.1; alpha: 0.1; gamma: 1; initial Q: 10.
Action masking: enabled (legal actions only). Choices use the mean of the two estimators.
`H` = hit; `S` = stand; `D` = double; `P` = split; `A` = dealer ace.
`?` = unseen legal choices or an untried legal choice leads/ties; `*` = incomplete coverage in either estimator; `/` = ties.
Unmarked choices are not proof of convergence or optimality.
These matrices apply only to a single unsplit hand: no completed or pending hands.
Split-context states: 1163426. Their exact contexts and choices are in the CSV; do not reuse these matrices for split children.

## Count <1; double no; split no; max hands 4

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard 6 | H | S | S* | H | S* | H | H* | H* | S | S* |
| Hard 7 | H | S | H | H | H | H | H | H | H | H |
| Hard 8 | H | H | H | H | H | H | H | H | H | H |
| Hard 9 | H | H | H | H | H | H | H | H | H | H |
| Hard 10 | H | H | H | H | H | H | H | H | H | H |
| Hard 11 | H | H | H | H | H | H | H | H | H | H |
| Hard 12 | H | H | H | S | H | S | H | H | H | H |
| Hard 13 | H | H | H | H | S | H | H | H | H | H |
| Hard 14 | S | H | S | S | S | H | H | H | H | H |
| Hard 15 | S | H | S | H | S | H | S | H | S | H |
| Hard 16 | S | S | S | S | S | S | H | H | H | H |
| Hard 17 | S | S | S | S | H | S | S | S | S | S |
| Hard 18 | S | S | S | S | S | S | S | S | S | S |
| Hard 19 | S | S | S | S | S | S | S | S | S | S |
| Hard 20 | S | S | S | S | S | S | S | S | S | S |
| Hard 21 | S | S | S | S | S | S | S | S | S | S |
| Soft 13 | H* | H* | S | S | H | S | H | H | H | H |
| Soft 14 | H | H | H | H | H | H | H | H | H | H |
| Soft 15 | H | H | H | H | H | H | H | H | H | H |
| Soft 16 | H | H | H | H | H | H | H | H | H | H |
| Soft 17 | H | H | H | H | H | H | H | H | H | H |
| Soft 18 | S | S | S | H | S | S | H | H | H | H |
| Soft 19 | S | S | S | S | S | S | S | S | S | S |
| Soft 20 | S | S | S | S | S | S | S | S | S | S |
| Soft 21 | S | S | S | S | S | S | S | S | S | S |
| Soft 21 (blackjack) | S | S | S | S | S | S | S | S | S | S |

## Count <1; double yes; split no; max hands 4

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard 5 | H | H | H | S | H | H | H | H | H | H |
| Hard 6 | H | H | H | H | H | H | H | H | H | H |
| Hard 7 | H | H | H | H | D | H | H | H | H | H |
| Hard 8 (five three) | H | H | H | H | H | H | H | H | H | H |
| Hard 8 (six two) | H | H | H | H | H | H | H | H | H | H |
| Hard 9 | H | H | H | H | D | D | H | H | H | H |
| Hard 10 | D | H | H | H | D | H | H | H | H | H |
| Hard 11 | H | H | H | H | H | H | H | H | H | H |
| Hard 12 | H | H | S | H | H | H | H | H | H | H |
| Hard 13 | H | H | H | S | S | H | H | H | S | H |
| Hard 14 | H | H | S | S | S | H | H | H | H | H |
| Hard 15 | H | S | H | S | S | H | H | S | H | H |
| Hard 16 | H | S | S | S | S | H | S | H | S | H |
| Hard 17 | S | S | S | S | S | S | S | S | S | S |
| Hard 18 | S | S | S | S | S | S | S | S | S | S |
| Hard 19 | S | S | S | S | S | S | S | S | S | S |
| Soft 13 | H | H | H | H | H | H | H | H | H | H |
| Soft 14 | H | H | H | H | H | D | H | H | H | H |
| Soft 15 | H | H | H | H | H | H | H | H | H | H |
| Soft 16 | H | H | H | H | H | H | H | H | H | H |
| Soft 17 | H | S | H | D | H | H | H | H | H | H |
| Soft 18 | H | H | S | S | H | S | S | H | H | H |
| Soft 19 | S | S | S | S | S | S | S | S | H | S |
| Soft 20 | S | S | S | S | S | S | S | S | S | S |

## Count <1; double yes; split yes; max hands 4

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard 4 (two two) | P | P | P | P | P | P | P | P | P | P |
| Hard 6 (three three) | P | P | P | P | P | P | P | P | P | P |
| Hard 8 (four four) | P | P | P | P | P | P | P | P | P | P |
| Hard 10 (five five) | P | P | P | P | P | P | P | P | P | P |
| Hard 12 (six six) | P | P | P | P | P | P | P | P | P | P |
| Hard 14 (seven seven) | P | P | P | P | P | P | P | P | P | P |
| Hard 16 (eight eight) | P | P | P | P | P | P | P | P | P | P |
| Hard 18 (nine nine) | P | P | P | P | P | P | P | P | P | P |
| Hard 20 (ten ten) | P | P | P | P | P | P | P | P | P | P |
| Soft 12 (ace ace) | P | P | P | P | P | P | P | P | P | P |

## Count >0; double no; split no; max hands 4

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard 6 | S | S | H | H | H | H | S | S | S | S |
| Hard 7 | H | H | H | H | H | H | H | H | H | H |
| Hard 8 | H | H | H | H | H | H | H | H | H | H |
| Hard 9 | H | H | H | H | H | H | H | H | H | H |
| Hard 10 | H | H | H | H | H | H | H | H | H | H |
| Hard 11 | H | H | H | H | H | H | H | H | H | H |
| Hard 12 | S | S | H | S | H | H | H | H | H | H |
| Hard 13 | H | H | S | S | H | H | H | H | H | H |
| Hard 14 | H | H | H | S | S | H | H | H | H | H |
| Hard 15 | S | S | S | S | S | H | H | H | H | H |
| Hard 16 | S | H | S | S | S | H | S | S | S | H |
| Hard 17 | S | S | S | S | S | S | S | S | S | S |
| Hard 18 | S | S | S | S | S | S | S | S | S | S |
| Hard 19 | S | S | S | S | S | S | S | S | S | S |
| Hard 20 | S | S | S | S | S | S | S | S | S | S |
| Hard 21 | S | S | S | S | S | S | S | S | S | S |
| Soft 13 | H | S* | S | H* | H* | H | H | S* | H | H |
| Soft 14 | H | H | H | H | H | H | H | H | H | H |
| Soft 15 | H | H | H | H | H | H | H | H | H | H |
| Soft 16 | H | H | H | H | H | H | S | H | H | H |
| Soft 17 | H | H | H | S | S | H | H | H | H | H |
| Soft 18 | H | S | H | S | S | S | H | H | H | S |
| Soft 19 | S | S | S | S | H | S | S | S | S | S |
| Soft 20 | S | S | S | S | S | S | S | S | S | S |
| Soft 21 | S | S | S | S | S | S | S | S | S | S |
| Soft 21 (blackjack) | S | S | S | S | S | S | S | S | S | S |

## Count >0; double yes; split no; max hands 4

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard 5 | H | H | H | H | H | H | H | H | H | H |
| Hard 6 | H | H | H | H | H | H | H | H | H | H |
| Hard 7 | H | H | H | H | H | H | H | H | H | H |
| Hard 8 (five three) | H | H | H | H | H | H | H | H | H | H |
| Hard 8 (six two) | H | H | H | H | H | D | H | H | H | H |
| Hard 9 | H | H | H | H | H | H | H | H | H | H |
| Hard 10 | H | H | H | H | D | D | D | H | H | H |
| Hard 11 | H | H | H | H | D | H | H | H | H | H |
| Hard 12 | H | H | S | S | S | H | H | H | H | H |
| Hard 13 | S | S | H | H | H | H | H | H | H | H |
| Hard 14 | H | S | S | S | S | H | H | H | H | H |
| Hard 15 | S | S | H | S | S | S | H | H | H | H |
| Hard 16 | S | S | S | S | S | H | H | H | H | H |
| Hard 17 | S | S | S | S | S | S | S | S | S | S |
| Hard 18 | S | S | S | S | S | S | S | S | S | S |
| Hard 19 | S | S | S | S | S | S | S | S | S | S |
| Soft 13 | H | H | D | H | H | H | H | H | H | H |
| Soft 14 | H | H | H | H | H | D | H | H | H | H |
| Soft 15 | H | H | H | H | H | H | H | H | H | H |
| Soft 16 | H | H | H | H | H | H | H | H | H | H |
| Soft 17 | H | H | H | H | S | H | H | H | H | H |
| Soft 18 | H | D | S | S | S | S | S | H | H | H |
| Soft 19 | S | S | S | S | S | S | S | S | H | S |
| Soft 20 | S | S | S | S | S | S | S | S | S | S |

## Count >0; double yes; split yes; max hands 4

| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hard 4 (two two) | P | P | P | P | P | P | P | P | P | P |
| Hard 6 (three three) | P | P | P | P | P | P | P | P | P | P |
| Hard 8 (four four) | P | P | P | P | P | P | P | P | P | P |
| Hard 10 (five five) | P | P | P | P | P | P | P | P | P | P |
| Hard 12 (six six) | P | P | P | P | P | P | P | P | P | P |
| Hard 14 (seven seven) | P | P | P | P | P | P | P | P | P | P |
| Hard 16 (eight eight) | P | P | P | P | P | P | P | P | P | P |
| Hard 18 (nine nine) | P | P | P | P | P | P | P | P | P | P |
| Hard 20 (ten ten) | P | P | P | P | P | P | P | P | P | P |
| Soft 12 (ace ace) | P | P | P | P | P | P | P | P | P | P |
