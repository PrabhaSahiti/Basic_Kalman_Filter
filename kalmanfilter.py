# kalman filter implementation
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys

#VARIABLES
#skip first row because it is initialized as initial reading
file = pd.read_csv('obj_pose-laser-radar-synthetic-input.txt', header=None, delim_whitespace = True, skiprows=1)

#initial readings
previous_time=1477010443000000/1000000.0 #covert from microseconds to seconds
state = np.array([[0.312242],[0.5803398],[0],[0]])

#initialize ground truth values and then initialize RMSE to compare with it
ground_truth=np.zeros([4,1])
RMSE=np.zeros([4,1])

P = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1000,0],[0,0,0,1000]])

A = np.array([[1.0,0,1.0,0],[0,1.0,0,1.0],[0,0,1.0,0],[0,0,0,1.0]])

I = np.identity(4)

H = np.array([[1.0,0,0,0],[0,1.0,0,0]])

present_state = np.zeros([2,1])

R = np.array([[0.0225,0],[0,0.0225]])

noise_ax= 5
noise_ay= 5
Q = np.zeros([4,4])

#FUNCTIONS

def predict():
    global state, P, Q
    state=np.matmul(A,state)
    A_transpose=np.transpose(A)
    P = np.add(np.matmul(A,np.matmul(P,A_transpose)), Q)

def update(present_state):
    global state,P
    intermediate_state = np.subtract(present_state,np.matmul(H,state))
    H_transpose = np.transpose(H)
    S = np.add(np.matmul(H, np.matmul(P,H_transpose)), R)
    S_inverse = inv(S)
    k = np.matmul(P,H_transpose)
    k = np.matmul(k, S_inverse)

    state = np.add(state, np.matmul(k,intermediate_state))
    P = np.matmul(np.subtract(I, np.matmul(k,H)),P)

def CalRMSE(estimations,ground_truth):
    rmse = np.zeros([4, 1])
    if (sys.getsizeof(estimations) != sys.getsizeof(ground_truth) or sys.getsizeof(estimations) == 0):
        print ('Invalid estimation or ground_truth data')
        return rmse
    rmse[0][0] =  np.sqrt(((estimations[0][0] - ground_truth[0][0]) ** 2).mean())
    rmse[1][0] =  np.sqrt(((estimations[1][0] - ground_truth[1][0]) ** 2).mean())
    rmse[2][0] =  np.sqrt(((estimations[2][0] - ground_truth[2][0]) ** 2).mean())
    rmse[3][0] =  np.sqrt(((estimations[3][0] - ground_truth[3][0]) ** 2).mean())
    print(rmse)
    return rmse



# MAIN LOOP
for i in range(len(file)):
    #calculate dt from timestamps
    measurement=file.iloc[i,:].values
    if measurement[0]=='L':
        current_time=measurement[3]/1000000.0
        delta_time=current_time - previous_time
        previous_time = current_time

        delta_time2=delta_time*delta_time
        delta_time3=delta_time2*delta_time
        delta_time4=delta_time3*delta_time

        #update A matrix
        A[0,2]=delta_time
        A[1,3]=delta_time

        #update Q matrix
        Q[0,0] = noise_ax*delta_time4/4
        Q[0,2] = noise_ax*delta_time3/2
        Q[1,1] = noise_ay*delta_time4/4
        Q[1,3] = noise_ay*delta_time3/2
        Q[2,0] = noise_ax*delta_time3/2
        Q[2,2] = noise_ax*delta_time2
        Q[3,1] = noise_ay*delta_time3/2
        Q[3,3] = noise_ay*delta_time2

        #update sensor readings
        present_state[0,0] = measurement[1]
        present_state[1,0] = measurement[2]

        #updating ground truths
        ground_truth[0] = measurement[4]
        ground_truth[1] = measurement[5]
        ground_truth[2] = measurement[6]
        ground_truth[3] = measurement[7]

        predict()
        update(present_state)
    print('iteration: ', i, 'state: ', state)
    RMSE=CalRMSE(state,ground_truth)
