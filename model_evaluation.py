"""
-V2.0 Updated 10/31/2017
 Separately calibrate model, allowing separate data transformations; 
 TODO:added **kwarg to function; add level forecast when data is transformed to difference
-V 1.0 Updated 10/22/2017
-V 0.0 Created 9/15/2016
-A replication of Cheng's Matlab code for developing and valuating point 
focecasting models with different measures. Resutls saved in .csv
-Steps: 
1. evaluate how the model fits to the data  
2. evaluate how accurate the model forecasts 
-Models include: 1. multi-step 2. multi-model 3. multi-dimentional
"""

import pandas as pd
import numpy as np
import data_reader
import statsmodels.api as sm
from statsmodels import regression
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt
from scipy import stats

def calibrate_ols(data):
    """OLS estimates have the same length as the original data"""
    ## 1. OLS
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]

    x1 = sm.add_constant(x)
    model_ols = regression.linear_model.OLS(y, x1).fit()
    #params_ols = model_ols.params
    #resids_ols = model_ols.resid
    fitted_values_ols = np.dot(x1, model_ols.params)

    return model_ols, fitted_values_ols

def calibrate_ecm(data):
    """ARIMA estimates have the same length as the original data"""
    ## 2. ECM
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]

    # dLnSt = a + z_t-1 + dLnSt_t-1 + dLnF_t + dLnF_t + epsi
    # --- step 1. residual from  lnSt = a1 + a2lnF, calculate resid given a1,a2
    x1 = sm.add_constant(x)
    model_ols = regression.linear_model.OLS(y, x1).fit()

    # --- step 2. Estimate the ECM
    x2 = sm.add_constant(np.column_stack((model_ols.resid[0:-1],
                                          y[0:-1],
                                          x[1:],
                                          x[0:-1])))

    model_ecm = regression.linear_model.OLS(y.iloc[1:], x2).fit()
    #params_ecm = model_ecm.params
    fitted_values_ecm = np.dot(x2, model_ecm.params)

    return model_ecm, fitted_values_ecm

def calibrate_arima(data, p=1, i=0, q=1):
    """ECM estimates have (data length - 1) due to auto-regression"""
    ## 3. ARIMA
    y = data.iloc[:, 0]
    #x = data.iloc[:, 1:]

    model_arima = ARIMA(y.values, order=(p, i, q))
    fit_arima = model_arima.fit()
    #params_arima = fit_arima.params
    fitted_values_arima = fit_arima.fittedvalues  # or use: fit_arima.predict(), model_arma.predict(params_arma)

    return model_arima, fitted_values_arima

def calibrate_var(data, order):
    """VAR estimates have (data length - 1) due to auto-regression"""
    ## 4. VAR
    #y4 = data.iloc[0:data_length-1, :].values
    model_var = VAR(data.values)
    fit_var = model_var.fit(order)
    fitted_values_var = fit_var.fittedvalues[:, 0] # or use: model_var.predict(fit_var.params)
    #lagData = (data.iloc[-1, :], model_ols.resid.iloc[-1])

    return model_var, fitted_values_var

def forecast_ols(params, independent_data):
    if len(independent_data) == 1:
        covariates = np.append(1., independent_data)
    else:
        covariates = sm.add_constant(independent_data)
    return np.dot(covariates, params)

def forecast_ecm(params, lag_resids, lag_y, x, lag_x):
    if len(lag_resids) == 1:
        covariates = np.column_stack([1., lag_resids, lag_y, x, lag_x])
    else:
        covariates = sm.add_constant(np.stack([lag_resids, lag_y, x, lag_x]))
    return np.dot(covariates, params)

def forecast_arima(fit, steps=1):
    data_len = len(fit.resid)
    # forecasts_arima = fit_arima.forecast()[0][0]
    return fit.predict(start=data_len, end=data_len + steps - 1)

def forecast_var(fit, develop_data, steps=1):
    # len_data = len(develop_data)
    # model_var.predict(params["VAR"], start=len_data, end=len_data + period )
    return fit.forecast(develop_data.values, steps)[:, 0]

class ModelTest():
    def __init__(self, data):
        self.data = data
        # self.window = window
        # self.freq = freq
        # self.step = step

        self.y = self.data.iloc[:, 0]
        self.x = self.data.iloc[:, 1:]

        self.results = None

    def estimate(self, window, freq, steps,
                 p=1, i=0, q=1, order=1):
        """
        :param window: specify window size for modeling  (should be > 30)
        :param freq: model calibrating frequency
        :param steps: look ahead (prediction) window
        :param p:
        :param i:
        :param q:
        :param var_order:
        :return:
        """

        for t in range(window, len(self.data) - steps):
            try:
                if t % freq == 0:
                    model_ols, _ = calibrate_ols(self.data.iloc[t-window:t, :])
                    model_ecm, _ = calibrate_ecm(self.data.iloc[t-window:t, :])
                    model_arima, _ = calibrate_arima(self.data.iloc[t-window:t, :], p, i, q)
                    model_var, _ = calibrate_var(self.data.iloc[t-window:t, :], order)

                sigma = np.std(self.data.iloc[t - window:t, 0])

                forecasts_ols = forecast_ols(model_ols.params, self.x.iloc[t:t + steps])

                resids = self.y.iloc[t:t + steps] - forecasts_ols
                lag_resids = np.append(model_ols.resid.iloc[-1], resids[:-1])
                forecasts_ecm = forecast_ecm(model_ecm.params,
                                             lag_resids,
                                             self.y.iloc[t - 1:t + steps - 1],
                                             self.x.iloc[t:t + steps],
                                             self.x.iloc[t - 1:t + steps - 1])

                forecasts_arima = forecast_arima(model_arima.fit(), steps)
                forecasts_var = forecast_var(model_var.fit(order),
                                             self.data.iloc[t - window:t, :], steps)

                if self.results is None:
                    self.results = pd.DataFrame({"step": t + steps,
                                                 "actual_y": self.y.iloc[t - 1 + steps],
                                                 "vol": sigma,
                                                 "forecast_OLS": forecasts_ols,
                                                 "forecast_ECM": forecasts_ecm,
                                                 "forecast_ARIMA": forecasts_arima,
                                                 "forecast_VAR": forecasts_var},
                                                index=np.arange(1))
                else:
                    self.results = pd.concat([self.results,
                                              pd.DataFrame({"step": t + steps,
                                                            "actual_y": self.y.iloc[t - 1 + steps],
                                                            "vol": sigma,
                                                            "forecast_OLS": forecasts_ols,
                                                            "forecast_ECM": forecasts_ecm,
                                                            "forecast_ARIMA": forecasts_arima,
                                                            "forecast_VAR": forecasts_var},
                                                           index=np.arange(1))],
                                             ignore_index=True)
            except:
                pass

    def evaluate_forecast(self, result_path):
        ## performance metrics
        realized_vol = self.results.vol
        forecasted = self.results.loc[:, [name for name in self.results.columns if name.split("_")[0] == "forecast"]]
        forecasted.columns = [name.split("_")[1] for name in forecasted.columns]
        realized = self.results.actual_y
        # realized = np.matrix(np.tile(results.actual_y, (3, 1))).T
        #forecasted = forecasted.cumsum()
        #realized = realized.cumsum()

        err = - forecasted.sub(realized, axis="index")
        MSE = (err ** 2).mean()
        RMSE = np.sqrt(MSE)
        MAE = err.abs().mean()
        MAPE = err.divide(realized, axis="index").abs().mean()

        test_stats = pd.concat([pd.DataFrame(MSE).T,
                                pd.DataFrame(RMSE).T,
                                pd.DataFrame(MAE).T,
                                pd.DataFrame(MAPE).T])
        asymmetric = pd.Series(0, index=test_stats.columns)

        for i, name in enumerate(test_stats.columns):
            if (err.loc[:, name] < 0).max() == True:
                asymmetric.loc[name] = ((err.loc[:, name] > 0).sum() / (err.loc[:, name] < 0).sum()) > 1
            else:
                asymmetric.loc[name] = np.Inf

        test_stats = test_stats.append(asymmetric, ignore_index=True)
        test_stats = test_stats.T
        test_stats.columns = ["MSE", "RMSE", "MAE", "MAPE", "asymmetric"]
        test_stats.to_csv(result_path + 'test_statistics.csv')

        ## plot
        for i, name in enumerate(test_stats.index):
            figure1 = plt.figure(figsize=(12, 9))
            ax11 = figure1.add_subplot(311)
            #ax11.plot(pd.concat([forecasted.loc[:, name], realized], axis=1))
            ax11.plot(realized, color="blue")
            ax11.plot(forecasted.loc[:, name], color="darkorange")
            ax11.legend([name + " forecasted", "realized"], loc="best")
            ax11.grid()
            ax11.set_title("realized vs. forecasted")

            ax12 = figure1.add_subplot(312)
            #ax12.plot(pd.concat([err.loc[:, name], err.loc[:, name].abs()], axis=1))
            ax12.plot(err.loc[:, name], color="blue")
            ax12.plot(err.loc[:, name].abs(), color="darkorange")
            ax12.legend([name + " forecast error", name + " absolute error"], loc="best")
            ax12.grid()
            ax12.set_title("forecasting error")

            ax13 = figure1.add_subplot(313)
            #ax13.plot(pd.concat([realized_vol, err.loc[:, name]], axis=1))
            ax13.plot(err.loc[:, name], color="blue")
            ax13.legend(["volatility"], loc="upper left")

            ax14 = ax13.twinx()
            ax14.plot(realized_vol, color="darkorange")
            ax14.legend([name + " forecast error"], loc="upper center")
            ax14.grid()
            ax14.set_title("volatility v.s forecast error")

            plt.suptitle(name + " forecasting performance")
            figure1_name = result_path + str(i) + '.' + name + '.realized_vs_forecasted.jpg'
            figure1.savefig(figure1_name)

            figure2 = plt.figure(figsize=(12, 9))
            ax21 = figure2.add_subplot(211)
            # calculate quantiles for a sample data against quantiles of a specified (default Gaussian)
            # distribution, and returns the best-fit line for the data
            (quantiles, values), (slope, intercept, r) = stats.probplot(err.loc[:, name], dist='norm')
            # (r is sqrt of r-squared)
            ax21.plot(quantiles, values, 'ob')  # values are the quantiles assuming data follows normal
            # this is the same as sm.qqplot(err.loc[:, name]
            ax21.plot(quantiles, quantiles * slope + intercept, 'r')
            #ticks_perc = [1, 5, 10, 20, 50, 80, 90, 95, 99]  # define ticks
            #ticks_quan = [stats.norm.ppf(j / 100.) for j in ticks_perc]  # transfrom them from precentile to cumulative density
            #plt.yticks(ticks_quan, ticks_perc)  # assign new ticks
            #ax21.grid()
            ax21.set_xlabel("Theoretical Quantiles", fontsize=6)
            ax21.set_ylabel("Sample Quantiles", fontsize=6)
            ax21.set_title("normality plot")

            ax22 = figure2.add_subplot(212)
            ax22.hist(err.loc[:, name], color="darkblue")
            #ax22.grid()
            ax22.set_title("error distribution")
            plt.suptitle(name + " error distribution")
            figure2_name = result_path + str(i) + '.' + name + '.error_distribution.jpg'
            figure2.savefig(figure2_name)

def main():
    data_path = 'model_evaluation_framework/dataFiles/'
    file_name = 'leadlag.csv'
    result_path = 'model_evaluation_framework/resultFiles/'
    data = data_reader.read_csv(data_path+file_name)

    #log_returns = data.diff().dropna()
    model_test = ModelTest(data[:300])
    model_test.estimate(window=30, freq=10, steps=1, p=1, i=0, q=1)
    model_test.evaluate_forecast(result_path)

if __name__ == "__main__":
    main()