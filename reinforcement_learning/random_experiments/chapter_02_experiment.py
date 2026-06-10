import numpy as np

# Verify Winder's statement on page 45
times_reached_the_end = 0
n = 10000
for _ in range(n):
    x = 0
    while x >= 0:
        x += np.random.choice([-1, 1])
        if x >= 4:
            times_reached_the_end += 1
            x = -1

print(f"{times_reached_the_end/n:.0%}")
