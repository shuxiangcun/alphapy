# futures_forecast.py 
"""
First version (v01) created on 10/5/2016 by Linchang.

Replicating SP index futures strategy. 
Par 1 of SPY forecast model: futures forecast model.

Update in v02 (10/12): when defining the RB system, changed to use the training 
sample to define cases and build up the RB system first and then feed the forecast 
sample into it, from using the forecast sample to define cases and throw back into 
the training sample to define case_result, while case_result for the training 
sample was not defined.

Update in v03 (10/14): removed some "attributes from the FuturesForecastModel().
Redefined the create_case and rb_system functions to make the training and forecasting
steps more in order.

Update in v04 (11/16): Standardized syntax based on Pylint feedback
"""

# Useful members after groupby a dataframe
# size: size/frequency of each group
# ngroups: number of groups/distinct values of groupby variable
# get_group: return cases falling under certain group. specify value of vector. 
#            e.g. (1,-1) if group by 2 vars
# groups: return indexes of cases falling under certain group
# indices: return # indices of cases falling under certain group  

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression


class FuturesForecastModel:
    """Implements the future forecast model with input data.
    
    Steps:
        1. Input full data (including forecast period).
        2. Run regression (price vs. time) and calculate defined 10 variables.
        3. Split data (w/ calculated vars) into training and forecast samples.
           and define case numbers for the forecast sample.
        4. Define the rule-based system, and generate trading recommendations
           for the forecast sample cases.
    
    Attributes:
        price: data
        intermediate_vars: data frame w/ intermediate variables in calculation appended.
        sample: data frame w/ 10 derived variables, plus 2 indicator for up/down day.
        train: training sample
        forecast: forecast sample
        trained_case: case information (case_no and case_result) of training sample
        forecasted_case: case information of forecast sample.
    """

    def __init__(self, data, train=None, forecast=None):
        self.price = data
        self.intermediate_vars = None
        self.sample = None
        self.train = train
        self.forecast = forecast
        self.trained_case = None
        self.forecasted_case = None

    def derive_variables(self, window, freq):
        """Derive variables based on regression and relevant calculation.
        
        Args:
            window: regression calibration window, default is 14 as in paper.
            freq: frequency of model calibration, default is 1 as in paper.
        """
        
        length = len(self.price)
        window = window  # time window for FFM regression model
        freq = freq  # frequency of regression calibration
        
        sp = pd.Series(-1, index=self.price.index)
        # sp: Equals 1 when the slope of price trend is significantly positive
        sn = pd.Series(-1, index=self.price.index)
        # sn: Equals 1 when the slope of price trend is significantly negative 
        c_f = pd.Series(0.0, index=self.price.index)
        # c_f: forecast close from linear model using previous 14 close
        fo = pd.Series(0.0, index=self.price.index)
        # fo: forecast oscillator
        ma3 = pd.Series(0.0, index=self.price.index)
        # 3-day mover average of the forecast oscillator
        lu = pd.Series(-1, index=self.price.index)
        # equals 1 when the oscillator crosses upward over its ma3
        ld = pd.Series(-1, index=self.price.index)
        # equals 1 when the oscillator crosses downward over its ma3
        
        up_moment = pd.Series(0.0, index=self.price.index)
        # up-day moment, equal |close_t - close_t-1| if close_t > close_t-1 o.w. 0
        down_moment = pd.Series(0.0, index=self.price.index)
        # down-day moment, equal |close_t - close_t-1| if close_t < close_t-1 o.w. 0
        ud = pd.Series(-1, index=self.price.index)
        # equals 1 when the closing price of the index is up at the present day
        aud = pd.Series(-1, index=self.price.index)
        # equals 1 when the closing prices are either up or down consecutively 
        # for at least 3 days
        
        upd = pd.Series(0, index=self.price.index)
        # equals 1 when the closing price of next day exceeds present day
        dnd = pd.Series(0, index=self.price.index)
        # equals 1 when the closing price of next day is less than present day
        
        sd = pd.Series(0.0, index=self.price.index)
        # up-day moment over 14-days
        su = pd.Series(0.0, index=self.price.index)
        # down-day moment over 14-days
        rsi = pd.Series(0.0, index=self.price.index)
        # relative strength index
        rsi_h = pd.Series(0.0, index=self.price.index)
        # highest RSI over past 14 days (incl. current)
        rsi_l = pd.Series(0.0, index=self.price.index)
        # lowest RSI over past 14 days (incl. current)
        stoch_rsi = pd.Series(0.0, index=self.price.index)
        # stochastic RSI
        
        rsi1 = pd.Series(-1, index=self.price.index)
        # equals 1 when the stochastic RSI falls from 100
        rsi2 = pd.Series(-1, index=self.price.index)
        # equals 1 when the stochastic RSI rises from 0
        rsi3 = pd.Series(-1, index=self.price.index)
        # equals 1 when the stochastic RSI is greater than 90
        rsi4 = pd.Series(-1, index=self.price.index)
        # equals 1 when the stochastic RSI is less than 10
               
        x = sm.add_constant(range(1, window+1))  # prepare x for regression
        
        # below variables start at index window, since regression takes window data points to start
        for t in range(window, length):
            if t % freq == 0:
                y = self.price[(t - window):t].values
                # run regression and evaluate beta and p-value
                model = regression.linear_model.OLS(y, x).fit()
                if model.params[1] > 0 and model.pvalues[1] < 0.05:
                    sp[t] = 1 
                elif model.params[1] < 0 and model.pvalues[1] < 0.05:
                    sn[t] = 1 
                x1 = (1, window+1)  # prepare X for one-step forecast
                c_f[t] = np.dot(x1, model.params)  # forecast price using regression
                fo[t] = 100*(self.price[t] - c_f[t])/self.price[t]

        # below variables start at index window+2, since ma3 takes another 2 data points to start
        for t in range(window + 2, length):
            ma3[t] = (fo[t] + fo[t-1] + fo[t-2])/3
            if fo[t-1] < ma3[t-1] and fo[t] > ma3[t]: 
                lu[t] = 1  # fo cross upward over ma3
            elif fo[t-1] > ma3[t-1] and fo[t] < ma3[t]:
                ld[t] = 1  # fo cross downward over ma3
        
        # below variables start at index 1
        for t in range(1, length):
            if self.price[t] > self.price[t-1]:
                up_moment[t] = abs(self.price[t] - self.price[t-1])
                ud[t] = 1
            elif self.price[t] < self.price[t-1]:
                down_moment[t] = abs(self.price[t] - self.price[t-1])

        # below variables start at index 3
        for t in range(3, length):
            if ((self.price[t] > self.price[t-1] > self.price[t-2] > self.price[t-3]) or
                    (self.price[t] < self.price[t-1] < self.price[t-2] < self.price[t-3])):
                aud[t] = 1
                
        # below variables start at index 0 till index length - 1
        for t in range(0, length - 1):
            if self.price[t+1] > self.price[t]:
                upd[t] = 1  # equals 0 otherwise
            elif self.price[t+1] < self.price[t]:
                dnd[t] = 1  # equals 0 otherwise
        
        # below variables start at index window, since up_moment & down_moment takes
        # 1 data point to start, and RSI takes (window-1) to start
        # All three include time t value
        for t in range(window, length):
            su[t] = up_moment[t - window + 1:t + 1].sum()
            sd[t] = down_moment[t - window + 1:t + 1].sum()
            rsi[t] = 100 * su[t] / (su[t] + sd[t])
            '''corrected RSI formula from original paper'''
        
        # below variables start at index 2*window-1, since rsi_h and rsi_l take
        # another (window-1) data points to start
        # All three include time t value
        for t in range(2*window - 1, length):
            rsi_h[t] = max(rsi[t - window + 1:t + 1])
            rsi_l[t] = min(rsi[t - window + 1:t + 1])
            stoch_rsi[t] = (100 * (rsi[t] - rsi_l[t]) / (rsi_h[t] - rsi_l[t]))
        
        # below variables start at index 2*window-1, since stoch_rsi takes 2*window-1 data points to start
        for t in range(2*window - 1, length):
            if stoch_rsi[t-1] == 100.0 and stoch_rsi[t] < 100.0:
                rsi1[t] = 1
            elif stoch_rsi[t-1] == 0.0 and stoch_rsi[t] > 0.0:
                rsi2[t] = 1
            if stoch_rsi[t] > 90.0:
                rsi3[t] = 1
            elif stoch_rsi[t] < 10.0:
                rsi4[t] = 1
        
        # append calculated variables to price and define data frames
        self.intermediate_vars = pd.concat([self.price, c_f, fo, ma3, up_moment,
                                            down_moment, su, sd, rsi, rsi_h, rsi_l,
                                            stoch_rsi], axis=1).iloc[2*window - 1:, ]
        self.intermediate_vars.columns = ["close", "forec_close", "forecast_oscillator",
                                          "ma3", "up_moment", "down_moment", "su", "sd",
                                          "rsi", "rsi_h", "rsi_l", "stoch_rsi"]
        self.sample = pd.concat([self.price, sp, sn, lu, ld, ud, aud, upd, dnd, 
                                 rsi1, rsi2, rsi3, rsi4], axis=1).iloc[2*window - 1:, ]
        self.sample.columns = ["close", "sp", "sn", "lu", "ld", "ud", "aud",
                               "upd", "dnd", "rsi1", "rsi2", "rsi3", "rsi4"]
        
        return self.sample

    def split_sample(self, cutoff):
        """Split data into train vs. forecast.
        
         Args:
            cutoff: cutoff date to split data into training and forecast.
        """
        self.train = self.sample[:cutoff]
        self.forecast = self.sample[cutoff:][1:]  # forecast starts from the day after cutoff date

    def train_rb_system(self):
        """Define the rule-based system using training sample.
        
        Calculate the case_result for all the case observations in the training
        sample, i.e., build the RB system.
        """

        train_case_no = pd.Series(0, index=self.train.index)
        # case_no in training and forecast samples are independent with each other
        # they are not of much use but merely for counting and comparing purposes
        gb_train = self.train.groupby(["lu", "ld", "sp", "sn", "ud", "aud", "rsi1", "rsi2", "rsi3", "rsi4"])
        for i, key in enumerate(gb_train.indices.keys()):
            train_case_no.loc[gb_train.groups[key]] = i
        train_ncase = gb_train.ngroups
    
        train_case_result = pd.Series("", index=self.train.index)
        # store case_result for case observations in the training sample
        
        for i in range(train_ncase):
            case = self.train[train_case_no == i]
            if ((case.lu[0] == -1) & (case.ld[0] == -1) &
               (case.rsi1[0] == -1) & (case.rsi2[0] == -1)):
                train_case_result[case.index] = "Trigger_OFF"
            else:
                u1, u2, u3, u4, d1, d2, d3, d4 = (0.0,)*8
                if case.lu[0] == 1:
                    u1 = self.train.ix[(self.train.lu == case.lu[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "upd"].sum()
                    d1 = self.train.ix[(self.train.lu == case.lu[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "dnd"].sum()
                if case.ld[0] == 1:
                    u2 = self.train.ix[(self.train.ld == case.ld[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "upd"].sum()
                    d2 = self.train.ix[(self.train.ld == case.ld[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "dnd"].sum()
                if case.rsi1[0] == 1:
                    u3 = self.train.ix[(self.train.rsi1 == case.rsi1[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "upd"].sum()
                    d3 = self.train.ix[(self.train.rsi1 == case.rsi1[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "dnd"].sum()
                if case.rsi2[0] == 1:
                    u4 = self.train.ix[(self.train.rsi2 == case.rsi2[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "upd"].sum()
                    d4 = self.train.ix[(self.train.rsi2 == case.rsi2[0]) &
                                       (self.train.sp == case.sp[0]) &
                                       (self.train.sn == case.sn[0]) &
                                       (self.train.ud == case.ud[0]) &
                                       (self.train.aud == case.aud[0]) &
                                       (self.train.rsi3 == case.rsi3[0]) &
                                       (self.train.rsi4 == case.rsi4[0]), "dnd"].sum()
                u = u1 + u2 + u3 + u4
                d = d1 + d2 + d3 + d4
                
                if u == d == 0.0:
                    # This could happen it there is only one observation for this case,
                    # and the Close of next day does not change, i.e. no up-day or down-day.
                    # Assign it to be "Obvious_WAIT" by discretion.
                    train_case_result[case.index] = "Obvious_WAIT"
                elif (u/(u+d)) >= .55:
                    train_case_result[case.index] = "Obvious_LONG"
                elif (d/(u+d)) >= .55:
                    train_case_result[case.index] = "Obvious_SHORT"
                elif u == d != 0.0:
                    train_case_result[case.index] = "Obvious_WAIT"
                elif (.50 < (u/(u+d)) < .55) or (.45 < (u/(u+d)) < .50):
                    train_case_result[case.index] = "Non_Obvious"
        self.trained_case = pd.concat([train_case_no, train_case_result], axis=1)
        self.trained_case.columns = ["case_no", "case_result"]

    def run_rb_system(self):
        """Run self.forecast through the RB system.
    
        Use the RB system to calculate case_result (trading recommendations)
        for each new case observation.
        """
        
        forecast_case_no = pd.Series(0, index=self.forecast.index)
        gb_forecast = self.forecast.groupby(["lu", "ld", "sp", "sn", "ud", "aud", "rsi1", "rsi2", "rsi3", "rsi4"])
        for i, key in enumerate(gb_forecast.indices.keys()):
            forecast_case_no.loc[gb_forecast.groups[key]] = i
        forecast_ncase = gb_forecast.ngroups
        
        forecast_case_result = pd.Series("", index=self.forecast.index)
        for i in range(forecast_ncase):
            case1 = self.forecast[forecast_case_no == i]
            case2 = self.train[(self.train.lu == case1.lu[0]) &
                               (self.train.ld == case1.ld[0]) &
                               (self.train.rsi1 == case1.rsi1[0]) &
                               (self.train.rsi2 == case1.rsi2[0]) &
                               (self.train.sp == case1.sp[0]) &
                               (self.train.sn == case1.sn[0]) &
                               (self.train.ud == case1.ud[0]) &
                               (self.train.aud == case1.aud[0]) &
                               (self.train.rsi3 == case1.rsi3[0]) &
                               (self.train.rsi4 == case1.rsi4[0])]  # exact same case
            if case2.shape[0] != 0:
                forecast_case_result[case1.index] = self.trained_case.ix[case2.index, "case_result"][0]
            else:
                forecast_case_result[case1.index] = "Unobserved"

        self.forecasted_case = pd.concat([forecast_case_no, forecast_case_result], axis=1)
        self.forecasted_case.columns = ["case_no", "case_result"]

        return self.forecasted_case

    def evaluate_cases(self):
        print "Number of FFM training cases: %s\n" % (len(self.trained_case.case_no.unique()))
        print "FFM trading recommendations by category (training cases): \n%s\n" % (self.trained_case.groupby("case_result").size())
        print "Most recent FFM trading recommendations (training cases): \n%s\n" % (self.trained_case.tail())

        print "Number of FFM forecast cases: %s\n" % (len(self.forecasted_case.case_no.unique()))
        print "FFM trading recommendations by category (forecast cases): \n%s\n" % (self.forecasted_case.groupby("case_result").size())
        print "Most recent FFM trading recommendations (forecast cases): \n%s\n" % (self.forecasted_case.tail())
