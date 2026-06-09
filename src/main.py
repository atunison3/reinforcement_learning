import numpy as np


arr = np.array([0, 0])
max_val = np.max(arr)
for _ in range(10):
    print(np.random.choice(np.where(arr == max_val)[0]))

len(x := np.array([0, 1]))
print(x)
