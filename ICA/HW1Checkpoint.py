import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
import time
from matplotlib.animation import FuncAnimation  # Used Claude AI to figure out animation below

import dill
import os
import sys

def checkpointSave(name,data):
    file=open(str(name),"wb+")
    dill.dump(data,file)
    file.close()

def checkpointLoad(name,data):
    if os.path.exists(str(name)):
        print("\n Checkpoint Loading... \n")
        with open(str(name),'rb') as file:
            data=dill.load(file)
            print("\n Loaded Data: ",data,"\n")
    else:
        return data
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name=sys.argv[1]
    else:
        name=0
    data = 0
    data=checkpointLoad(name, data)
    while data<100:
        data+=1
        if data%1==0:
            checkpointSave(name, data)
        print("Data=",data)
        time.sleep(1)

def update(frame):
    line.set_ydata(y[frame])
    return line,

def main():
    start_time = time.time()  # begin measuring runtime

    # Divide simulation into grid in the x direction
    xmin = 0
    xmax = 10
    nx = 512
    dx = (xmax - xmin) / nx
    x = np.linspace(xmin, xmax, nx)
    
    # Divide time into discrete units
    tmin = 0
    tmax = 10
    nt = 1000  # smaller nt to take less time
    dt = (tmax - tmin) / nt
    
    # Initialize starting position as a simple pulse
    # 2D array idea from Claude AI for animation
    y = np.zeros((nt, nx))  # 2D array to capture positions in each timestep
    y[0,:] = np.exp(-(x - 5)**2)
    
    # Initialize velocity and acceleration to zero
    v = np.zeros((nt, nx))
    a = np.zeros((nt, nx))
    gamma = 1
    
    # Run the simulation of t timesteps
    for t in range(1, nt - 1):
        a[t, 0] = 0
        a[t, nx - 1] = 0
        for i in range(1, nx - 1):
            a[t, i] = gamma * (y[t - 1,i + 1] + y[t - 1,i - 1] - 2*y[t - 1, i]) / (dx**2)
        for i in range(nx):
            v[t, :] = v[t - 1, :] + a[t, :] * dt
            y[t, :] = y[t - 1, :] + v[t, :] * dt
    
    # Used Claude AI to figure out animation below
    # https://claude.ai/chat/
    # 1/26/25
    # Prompt: For this code:
            # # Divide simulation into grid in the x direction
            # xmin = 0
            # xmax = 10
            # nx = 512
            # dx = (xmax - xmin) / nx
            # x = np.linspace(xmin, xmax, nx)
            # # Divide time into discrete units
            # tmin = 0
            # tmax = 10
            # nt = 1000
            # dt = (tmax - tmin) / nt
            # times = np.linspace(tmin, tmax, nt, dtype=int)
            # # Initialize starting position as a simple pulse
            # y = np.exp(-(x - 5)2)
            # # Initialize velocity and acceleration to zero
            # v = 0 * x
            # a = 0 * x
            # gamma = 1
            # # Run the simulation of t timesteps
            # for t in times:
            #     plt.clf()
            #     a[0] = 0
            #     a[nx - 1] = 0
            #     for i in range(1, nx - 1):
            #         a[i] = gamma * (y[t + 1] + y[t - 1] - 2*y[t]) / (dx2)
            #     for i in range(nx):
            #        v[t] += a[t] * dt
            #         y[t] += v[t] * dt
            #     plt.plot(x, y)
            #     plt.draw()
            #     plt.show()
            # how do I make a gif of the resulting plots and download?
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x, y[0])
    ax.set_ylim((-1, 1))
    ax.set_title('Wave Simulation')
    
    ani = FuncAnimation(fig, update, frames=nt, interval=20, blit=True)
    
    # Save as GIF
    ani.save('wave_simulation.gif', writer='pillow')
    plt.close(fig)
    
    print("Runtime:", time.time() - start_time)  # print runtime