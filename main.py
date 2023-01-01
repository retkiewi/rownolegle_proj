import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from star import Star
from mpi4py import MPI
from math import ceil, log10


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
range_end = min((rank+1)*block_size, N)

stars = stars[range_start:range_end]

result = []

# print(len(stars), rank)
padding = int(log10(ITERS))
for i in range(ITERS):
    if rank == 0:
        progress = '-'*(50*i//ITERS)+'>'
        print(f'ITER {i:>5} [{progress:<50}]',  end='\r')
    # get the buffoer from prev
    attractions = np.zeros((len(stars),3))
    comm.send(stars, dest=(rank+1)%size, tag=1)
    comm.send(np.zeros((len(stars),3)), dest=(rank+1)%size, tag=3)

    for j in range(size//2):
        buffer_stars = comm.recv(source=(rank-1)%size, tag=1)
        buffer_attractions = comm.recv(source=(rank-1)%size, tag=3)

        attractions = attractions + [star.calculate_attraction(buffer_stars) for star in stars]

        #avoid calculating attractions twice when we an even number of threads 
        if size%2 != 0:
            buffer_attractions =  buffer_attractions + [star.calculate_attraction(stars) for star in buffer_stars]

        if j<size//2-1:
            comm.send(buffer_stars, dest=(rank+1)%size, tag=1)
            comm.send(buffer_attractions, dest=(rank+1)%size, tag=3)

    # exchange the attraction accumulator
    comm.send(buffer_attractions, dest=(rank-size//2)%size, tag=3)
    buffer_attractions = comm.recv(source=(rank+size//2)%size, tag=3)

    attractions = attractions + buffer_attractions

    if i%print_n==0:
        if rank == 0:
            curr_result = [(star.position, int(star.mass**(1/3))) for star in stars]
            for j in range(1, size):
                curr_result = curr_result + [(star.position, int(star.mass**(1/3))) for star in comm.recv(source=j, tag=2)]
            result.append(curr_result)
        else:
            comm.send(stars, dest=0, tag=2)
    
    for star, attraction in zip(stars, attractions):
        star.tick_velocity(attraction)
        star.tick_position()

if rank != 0: exit(0)

print()

def update_points(t):
    plt.gca().cla()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(-300, 300)
    for star in result[t]:
        ax.scatter(*star[0], s=star[1])

    
anim = animation.FuncAnimation(fig, update_points, frames=ITERS//print_n, interval=1)

anim.save('./result.gif', fps=30)
