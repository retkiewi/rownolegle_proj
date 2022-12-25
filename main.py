from matplotlib import pyplot as plt
from matplotlib import animation
from simulation import run_simulation

N = 10
ITERS = 600


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

result = run_simulation(N, ITERS)

def update_points(t):
    plt.gca().cla()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(-300, 300)
    for star in result[t]:
        ax.scatter(*star[0], s=star[1])

    
    
anim = animation.FuncAnimation(fig, update_points, frames=ITERS, interval=1)

anim.save('./result.gif', fps=30)
