import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from star import Star
from mpi4py import MPI
from math import ceil


ITERS = 40000
print_n = 100

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 2*size

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

if rank == 0:
    stars = [Star() for _ in range(N)]
else:
    stars = None

stars = comm.bcast(stars, root=0)

block_size = ceil(N/size)
range_start = rank*block_size
range_end = max((rank+1)*block_size, N)

stars = stars[range_start, range_end]

result = []

print(len(stars), rank)

for i in range(ITERS):
    # get the buffoer from prev
    attractions = np.zeros(len(stars))
    comm.send(stars, dest=(rank+1)%size, tag=1)

    for j in range(size-1):
        prev_stars = comm.recv(source=(rank-1)%size, tag=1)

        attractions = attractions + [star.calculate_attraction(prev_stars) for star in stars]
        # other_attractions = [star.calculate_attraction(stars) for star in prev_stars] + other_attractions

        if j<size-2:
            comm.send(prev_stars, dest=(rank+1)%size, tag=1)
    
    if rank == 0 and i%print_n==0:
        curr_result = [*stars]
        for j in range(1, size):
            curr_result = curr_result + comm.recv(source=j, tag=2)
        result.append(curr_result)
    else:
        comm.send(stars, dest=0, tag=2)
    

    for star, attraction in zip(stars, attractions):
        star.tick_velocity(attraction)
        star.tick_position()
    

# for i in range(ITERS):
#     for star in stars[range_start:range_end]:
#         star.tick_velocity(stars)
#     for star in stars[range_start:range_end]:
#         star.tick_position()
#     comm.send(stars, dest=(rank+1)%size, tag=1)
#     comm.send(stars, dest=(rank-2)%size, tag=2)
#     prev_stars = comm.recv(source=(rank-1)%size, tag=1)
#     next_stars = comm.recv(source=(rank+2)%size, tag=2)
#     stars[:range_start] = prev_stars[:range_start]
#     stars[range_end:] = prev_stars[range_end:]
#     next_block_end = (range_end+block_size)%N
#     if next_block_end == 0: next_block_end = 48
#     stars[range_end%N:next_block_end] = next_stars[range_end%N:next_block_end]
#     next_block_end = (range_end+block_size*2)%N
#     if next_block_end == 0: next_block_end = 48
#     stars[(range_end+block_size)%N:(range_end+block_size*2)%N] = next_stars[(range_end+block_size)%N:(range_end+block_size*2)%N]
#     if rank == 0 and i%print_n==0:
#         result.append([(star.position, int(star.mass**(1/3))) for star in stars])

if rank != 0: exit(0)


def update_points(t):
    plt.gca().cla()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(-300, 300)
    for star in result[t]:
        ax.scatter(*star[0], s=star[1])

    
anim = animation.FuncAnimation(fig, update_points, frames=ITERS//print_n, interval=1)

anim.save('./result.gif', fps=30)
