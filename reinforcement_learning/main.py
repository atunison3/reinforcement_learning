import numpy as np

for _ in range(100000):
    x = np.random.random()
    if x < 0 or x > 1:
        print("No")
        break

print("Finished")
