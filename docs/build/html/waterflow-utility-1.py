import matplotlib.pyplot as plt
from waterflow.utility.spacing import spacing, biasedspacing

fig, [ax1, ax2] = plt.subplots(figsize=(9, 9), nrows=2, ncols=1, sharex=True)
powers = list(range(1, 7))
maxdists = [0.10 + 0.03 * i for i in range(len(powers))]
for i, j in zip(powers, maxdists):
    a1, a2 = biasedspacing(11, i), biasedspacing(11, powers[-1], maxdist=j)
    ax1.scatter(a1, [i for _ in range(len(a1))], marker='*', color='blue')
    ax2.scatter(a2, [j for _ in range(len(a2))], marker='*', color='blue')

ax1.set_title('Nodal discretization of 1D biasedspacing function')
ax1.set_yticks(powers)
ax1.set_ylabel('variable power, no maxdist')
ax1.grid()
ax2.set_yticks(maxdists)
ax2.set_xlabel('Distance (x)')
ax2.set_ylabel(f'Variable maxdist, fixed power({powers[-1]})')
ax2.grid()
plt.show()