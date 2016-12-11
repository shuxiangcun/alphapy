# kalman_filter_with_EM.py
"""
sss
"""

import matplotlib.pyplot as plt
import pandas as pd
import math
import data_reader
from pykalman import KalmanFilter
#from kalman_filter import KalmanFilter


def double_exp_MA(series):
    N = len(series)
    _lambda = 1 - 2 / (float(N) + 1)
    alpha = 1 - _lambda
    a1 = pd.Series(0.0, index=series.index)
    a2 = pd.Series(0.0, index=series.index)
    a1[0] = series[0]
    a2[0] = series[0]

    for i in range(1, N):
        a1[i] = _lambda * a1[i - 1] + alpha * series[i]
        a2[i] = _lambda * a2[i - 1] + alpha * a1[i]
    return 2 * a1 - a2

def main():
    # 0. Read in data
    filename1 = "data/GE.Last.txt"
    filename2 = "data/PWR.Last.txt"

    data1 = data_reader.read_ninja_txt(filename1)
    data2 = data_reader.read_ninja_txt(filename2)

    # 1. Use only Kalman to estimate past PWR price (state)
    H = 1.1

    # The initial setup of observation and transition matrices has big impact on the outcome, because this gives a very
    # strong initial opinion on the relationship b/w measurement and the latent state, and b/w the latent states themselves

    measurements = pd.DataFrame(data1.Close).apply(math.log, axis=1)
    initial_state_mean = measurements[0]/H

    kf = KalmanFilter(transition_matrices=0.9,
                      observation_matrices=H,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)

    state_means, state_covariance = kf.filter(measurements)
    state_means = pd.DataFrame(state_means, index=data2.index)
    actual_states = pd.DataFrame(data2.Close).apply(math.log, axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(state_means)
    plt.plot(actual_states)
    plt_title = 'Kalman filter estimate of PWR price (H = %s)' % H
    plt.title(plt_title)
    plt.legend(['Kalman Estimate PWR', 'Actual PWR'], loc="best")
    plt.xlabel('Time')
    plt.ylabel('Log Price')
    output_file = 'result/kalman/1.kalman_vs_actual_PWR_H_eq_%s.pdf' % H
    fig.savefig(output_file)
    plt.close(fig)

    # 2. Apply double exponential moving average on the observation matrix (H)
    actual_H = measurements/actual_states
    demv_H = double_exp_MA(actual_H)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    plt.plot(actual_H)
    plt.plot(demv_H)
    plt.title('Double Exponential Moving Average of beta (H)')
    plt.legend(['Actual H', 'DEMV H'], loc="best")
    plt.xlabel('Time')
    output_file = 'result/kalman/2.actual_vs_demv_beta.pdf'
    fig2.savefig(output_file)
    plt.close(fig2)

    # 3. Use Kalman and EM to estimate past PWR price
    kf = kf.em(measurements, n_iter=100, em_vars = ['transition_matrices',
    'observation_matrices', 'transition_covariance', 'observation_covariance', 
    'initial_state_mean', 'initial_state_covariance'])
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
    filtered_state_means = pd.DataFrame(filtered_state_means, index=data2.index)
    smoothed_state_means = pd.DataFrame(smoothed_state_means, index=data2.index)

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    plt.plot(filtered_state_means)
    #plt.plot(smoothed_state_means)
    plt.plot(actual_states)
    plt_title = 'Kalman filter (with EM) estimate of PWR price (H = %s)' % H
    plt.title(plt_title)
    plt.legend(['Kalman filtered PWR', 'Actual PWR'], loc="best")
    plt.xlabel('Time')
    plt.ylabel('Log Price')
    output_file = 'result/kalman/3.kalman_nd_em_vs_actual_PWR_H_eq_%s.pdf' % H
    fig3.savefig(output_file)
    plt.close(fig3)

    # 4. Treat the observation matrix as latent state, and Kalman and EM to estimate its past values

    measurements = pd.DataFrame(data1.Close).apply(math.log, axis=1)
    initial_state_mean_beta = 1.1
    H_beta = measurements[1] / initial_state_mean_beta

    kf_beta = KalmanFilter(transition_matrices=0.9,
                           observation_matrices=H_beta,
                           initial_state_mean=initial_state_mean_beta,
                           initial_state_covariance=1,
                           observation_covariance=1,
                           transition_covariance=0.01)

    kf_beta = kf_beta.em(measurements, n_iter=10)
    '''
    TODO:
    add transition_matrices and observation_matrices to EM_Vars
    '''
    filtered_state_means_beta, _ = kf_beta.filter(measurements)
    filtered_state_means_beta = pd.DataFrame(filtered_state_means_beta, index=data2.index)
    actual_states_beta = actual_H

    fig4, ax4 = plt.subplots(figsize=(10, 7))
    plt.plot(filtered_state_means_beta)
    plt.plot(actual_states_beta)
    plt_title = 'Kalman filter (with EM) estimate of beta (H = %s)' % H_beta
    plt.title(plt_title)
    plt.legend(['Kalman Estimate beta', 'Actual beta'], loc="best")
    plt.xlabel('Time')
    plt.ylabel('Log Price')
    output_file = 'result/kalman/4.kalman_nd_em_vs_actual_beta_H_eq_%s.pdf' % H
    fig4.savefig(output_file)
    plt.close(fig4)

if __name__ == '__main__':
    main()
