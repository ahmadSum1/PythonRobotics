"""

Extended kalman filter (EKF) localization for asynchronus sensors

author: Sakib Ahmed (@ahmadSum1)
modified upon Atsushi Sakai (@Atsushi_twi) 's implementation

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math
# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

from utils.angle import rot_mat_2d

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    0.5  # variance of velocity
    ]) ** 2  # predict state covariance
R = np.diag([5.0, 5.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
ODOM_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

f_odom = 100
f_gps = 1

DT = 1/f_odom  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True

# genarates the trajectory
def calc_input(ccw=True):
    v = 1.0  # [m/s]
    if ccw:
        yawrate = 0.1  # [rad/s]
    else:
        yawrate = -0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation_IMU_Odom(xdr, u):
    
    # add noise to input
    ud = u + ODOM_NOISE @ np.random.normal( size=(2, 1))  #gaussian noise

    xdr = motion_model(xdr, ud)  #dead rekoning position

    return xdr, ud

def observation_z(xTrue):
    
    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.normal( size=(2, 1))

    return z



def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH



def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rot_mat_2d(angle) @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    z = np.zeros((2, 1))

    # while SIM_TIME >= time:
    #     time += DT
    for k in range(int(SIM_TIME/DT)):

        #calculate/update real trjectory/pose every 10ms
        u = calc_input() #* np.array([[1],[0]]) #real movement

        if k>SIM_TIME/DT/2: #change direction
            u = calc_input(ccw=False)
            
        xTrue = motion_model(xTrue, u) #true position

        xDR, ud = observation_IMU_Odom(xDR, u)  # ud = measured movement/input; xDR = dead reckoning position, nothing to do with ekf, just for comparison;


        #  Predict
        xPred = motion_model(xEst, ud)
        jF = jacob_f(xEst, ud)
        PPred = jF @ PEst @ jF.T + Q                        # P is covariace matrix of the state,
        
        # at GPS measurement freq
        if k%int(f_odom/f_gps) == 0:
            z = observation_z(xTrue) #GPS data
            #  Update
            jH = jacob_h()
            zPred = observation_model(xPred)
            y = z - zPred
            S = jH @ PPred @ jH.T + R
            K = PPred @ jH.T @ np.linalg.inv(S)
        
            xEst = xPred + K @ y
            PEst = (np.eye(len(xEst)) - K @ jH) @ PPred         # P is covariace matrix of the state,
        else:
            xEst = xPred
            PEst = PPred


        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g", label="GPS")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b", label="Ground Truth")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k", label="Dead Reckoning")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), ":r", label="EKF")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(  "Time = {:2f}s".format(k*DT),
                        fontsize='small',
                        loc='left')
            plt.pause(0.001)
    plt.show()
   
    # input("\n press enter to exit\n")


if __name__ == '__main__':
    main()
