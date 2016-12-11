# kalman_filter.py

import numpy  as np
import pandas as pd

class KKalmanFilter(object):
    def __init__(self, Y=None, X0=None, P0=None, F=None, H=None, Q=None, R=None, B=None, b=None):
        '''Create a Kalman filter by setting.

        Assume one dimension first

        Params:
            dim_x: dimension of state variables.
            dim_y: dimension of measurement inputs (observation).
            dim_u: (optional) dimension of the control input (offset). Default of 0 indicates it is not used.
        '''



        self.Y = Y # observation matrix
        self.X0 = X0  # initial state mean
        # #ndim_x = X.shape[0] if X.ndim > 1 else 1
        #ndim_y = Y.shape[0] if Y.ndim > 1 else 1
        self.P0 = P0 # initial state covariance
        self.F = F # transition matrix
        self.H = H # observation matrix
        self.Q = Q # transition covariance
        self.R = R # observation covariance

        self.B = B # offset transition matrix
        self.b = b # offset

        #self._I = np.eye(ndim_x) # identity matrix

    def filter(self, Y):
        self.X_est = pd.Series(0.0, index=Y.index)
        self.P_est = pd.Series(0.0, index=Y.index)
        Y_est = pd.Series(0.0, index=Y.index)
        _error = pd.Series(0.0, index=Y.index)
        S = pd.Series(0.0, index=Y.index)
        kal_gain = pd.Series(0.0, index=Y.index)

        _offset = 0 if (self.B==None or self.b==None) else self.B * self.b

        for t in range(len(Y)):
            if t == 0:
                # 1. initiation
                self.X_est[t] = self.X0
                self.P_est[t] = self.P0
            else:
                # 2. calculate X_t|t-1 and P_t|t-1
                self.X_est[t] = self.F * self.X_est[t-1] + _offset
                self.P_est[t] = self.F * self.P_est[t-1] * self.F + self.Q

            # 3. calculate forecast error
            Y_est[t] = _offset + self.H * self.X_est[t]
            _error[t] = Y[t] - Y_est[t]

            # 4. update to X_t|t and P_t|t
            S[t] = self.H * self.P_est[t] * self.H + self.R  # predicted observation variance
            kal_gain[t] = self.P_est[t] * self.H * S[t]**(-1)
            self.X_est[t] = self.X_est[t] + kal_gain[t] * _error[t]
            self.P_est[t] = self.P_est[t] - kal_gain[t] * self.H * self.P_est[t]

        return self.X_est, self.P_est