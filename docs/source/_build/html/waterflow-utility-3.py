import matplotlib.pyplot as plt
from waterflow.utility.spacing import spacing

x, y = spacing(11, 10, 11, 10, linear=False, loc=[(4, 5)], power=2, weight=4)

fig, ax = plt.subplots(figsize=(9, 9))
for i in y:
    ax.scatter(x, [i for _ in range(len(x))], marker='*', color='blue')

ax.set_title('Nodal discretization of 2D spacing function')
ax.set_xlabel('Distance (x)')
ax.set_ylabel('Distance (y)')
ax.grid()