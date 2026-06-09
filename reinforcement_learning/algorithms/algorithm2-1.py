"""
Algorithm 2-1. epsilon greedy algorithm
1. INPUT: an exploration probability 0 <= epsilon <= 1
2. Initiatize r_avg(a) <- 0, N(a) <- 0, for each a in A
3. loop for ever:
    4.
        a. choose to explore or exploit with probabilities of (epsilon, 1 - epsilon).
        b. assign action argmax r_avg(a) if exploiting or random(a) it exploring
    5. Present action a in the environment and receive reward r
    6. N(a) <- N(a) + r
    7. r_avg(a) <- r_avg(a) + 1/N(a)(r-r_avg(a))
"""

import numpy as np

# Game: choose $1 prize behind door 1 or door 2 with prize being randomly placed behind either door with probabilities (0.25, and 0,75)

# Step 1 - assign epsilon between 0 and 1
epsilon = 0.5

# Step 2 - Initialize reward states
r_avg = [0, 0]  # [left door 1 or right door 2]
n_a = [0, 0]

n_e = 0

# Step 3 - Loop for ever (jk for 10,000 loops)
for _ in range(10000):

    # Step 4 - Assign action
    if np.random.binomial(1, epsilon) == 1:
        a = np.random.choice([0, 1])
    else:
        if r_avg[0] == r_avg[1]:
            a = np.random.choice([0, 1])
        else:
            a = int(np.argmax(r_avg))

    # Step 5 - Receive reward
    if a == 0:
        n_e += 1
    if a == 0:
        r = np.random.binomial(1, 0.25)
    else:
        r = np.random.binomial(1, 0.75)

    # Step 6 - Increment counter
    n_a[a] += 1

    # Step 7 - Calculate the average reward
    r_avg[a] = r_avg[a] + a / n_a[a] * (r - r_avg[a])

# Print results
print(f"The agent's choice probabilities are: {r_avg[0]} - {r_avg[1]}")
