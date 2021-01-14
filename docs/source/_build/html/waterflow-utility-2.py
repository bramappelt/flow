import matplotlib.pyplot as plt
from waterflow.utility.spacing import spacing

fig, ax = plt.subplots(figsize=(9.0, 2))
fig.suptitle('Nodal discretization of 1D spacing function')
x, _ = spacing(10, 10, linear=False, loc=[4, 7], power=1, weight=3)
ax.scatter(x, [0 for i in range(len(x))], marker='*', color='blue')
ax.set_xlabel('Distance (x)')
ax.grid()