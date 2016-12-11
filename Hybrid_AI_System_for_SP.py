# hybrid_ai_system_for_sp_v03.py
"""
First version created on 9/28/2016 by Linchang

Replicating strategy as in:
Forecasting S&P 500 stock index futures with a hybrid AI system, Tsaih, Hsu,
Lai, 1998.

Steps: 

Update in v02 (10/5): standardize the module. Move code defining FFM to separate file.
Update in v03 (10/15): change to accommodate FFM v03 and EFFM v01.
Update in v04 (11/16): Standardized syntax based on Pylint feedback, and added in strategy evaluation
"""


def main():
    import data_reader
    import pandas as pd
    import matplotlib.pyplot as plt
    from futures_forecast import FuturesForecastModel
    from extended_futures_forecast import ExtendedFuturesForecastModel

    # Do following in case of reloading modules (in development)
    # import sys
    # del sys.modules[ExtendedFuturesForecastModel.__module__]


    #######################
    # 1. Read in data
    filename = "data/SPY.Last.txt"
    data = data_reader.read_ninja_txt(filename)


    #######################
    # 2. Train and run FFM
    strategy1 = FuturesForecastModel(data.Close)
    strategy1.derive_variables(window=14, freq=1)
    strategy1.split_sample('2009-12-31')  # forecast starts from 2010-1-1
    strategy1.train_rb_system()
    forecasted_case = strategy1.run_rb_system()

    #######################
    # 3. Train and run EFFM

    #  only run the EFFM on Non Obvious cases
    non_obvious_forecast = strategy1.forecast[forecasted_case.case_result == "Non_Obvious"]

    # 3.1 Train and run BP
    '''Set leas_false temporarily to be high to allow convergence.
       TO-DO: improvement on the algo to gain better performance.
    '''
    strategy2 = ExtendedFuturesForecastModel(strategy1.train, strategy1.trained_case, non_obvious_forecast)
    strategy2.create_train_set()
    strategy2.train_lu(method="bp", num_hidden=4, num_train=10000, least_false=2)
    strategy2.train_ld(method="bp", num_hidden=4, num_train=10000, least_false=1.8)
    strategy2.train_rsi1(method="bp", num_hidden=4, num_train=10000, least_false=1.8)
    strategy2.train_rsi2(method="bp", num_hidden=4, num_train=10000, least_false=1.5)
    updated_forecast_case_bp = strategy2.run_nn()


    '''
    # 3.2 Train and run Perceptron
    strategy3 = ExtendedFuturesForecastModel(strategy1.train, strategy1.trained_case, non_obvious_forecast)
    strategy3.create_train_set()
    strategy3.train_lu(method="pn", num_train=10000, least_false=0.2)
    strategy3.train_ld(method="pn", num_train=10000, least_false=0.2)
    strategy3.train_rsi1(method="pn", num_train=10000, least_false=0.5)
    strategy3.train_rsi2(method="pn", num_train=10000, least_false=0.3)
    updated_forecast_case_pn = strategy3.run_nn()

    # evaluate result
    strategy3.evaluate_cases()
    '''

    #######################
    # 4. Test performance

    # 4.1 FFM prediction success rate
    strategy1.evaluate_cases()  # evaluate result
    ffm_forecast = forecasted_case
    ffm_success_rate = 0.0
    ffm_test_forecast = ffm_forecast[ffm_forecast["case_result"].isin(["Obvious_LONG", "Obvious_SHORT"])]

    for index in ffm_test_forecast.index:
        if ((ffm_test_forecast.ix[index, "case_result"] == "Obvious_LONG" and
                     strategy1.forecast.ix[index, "upd"] == 1) or
           (ffm_test_forecast.ix[index, "case_result"] == "Obvious_SHORT" and
                     strategy1.forecast.ix[index, "dnd"] == 1)):
            ffm_success_rate += 1

    ffm_success_rate /= len(ffm_test_forecast)

    print "\nFFM success rate is: %s\n" % ffm_success_rate

    # 4.2 EFFM prediction success rate
    strategy2.evaluate_cases()  # evaluate result
    effm_forecast = updated_forecast_case_bp
    effm_success_rate = 0.0
    effm_test_forecast = effm_forecast[effm_forecast["case_result"].isin(["LONG", "SHORT"])]

    for index in effm_test_forecast.index:
        if ((effm_test_forecast.ix[index, "case_result"] == "LONG" and strategy2.forecast.ix[index, "upd"] == 1) or
             (effm_test_forecast.ix[index, "case_result"] == "SHORT" and strategy2.forecast.ix[index, "dnd"] == 1)):
            effm_success_rate += 1

    if len(effm_test_forecast) ==0:
        effm_success_rate = 0
    else:
        effm_success_rate /= len(effm_test_forecast)

    print "\nEFFM success rate is: %s\n" % effm_success_rate

    # 4.3 Hybrid (FFM + EFFM) prediction success rate
    hybrid_success_rate = 0.0
    hybrid_forecast = forecasted_case
    for index in hybrid_forecast.index:
        if hybrid_forecast.ix[index, "case_result"] == "Non_Obvious":
            hybrid_forecast.ix[index, "case_result"] = updated_forecast_case_bp.ix[index, "case_result"]

    hybrid_test_forecast = hybrid_forecast[hybrid_forecast.case_result.isin(["Obvious_LONG", "LONG",
                                                                             "Obvious_SHORT", "SHORT"])]

    for index in hybrid_test_forecast.index:
        if (((hybrid_test_forecast.ix[index, "case_result"] == "Obvious_LONG" or
              hybrid_test_forecast.ix[index, "case_result"] == "LONG") and strategy1.forecast.ix[index, "upd"] == 1) or
            ((hybrid_test_forecast.ix[index, "case_result"] == "Obvious_SHORT" or
              hybrid_test_forecast.ix[index, "case_result"] == "SHORT") and strategy1.forecast.ix[index, "dnd"] == 1)):
            hybrid_success_rate += 1

    hybrid_success_rate /= len(hybrid_test_forecast)
    print "\nHybrid success rate is: %s\n" % hybrid_success_rate

    # 4.4 Hybrid system strategy return
    buy_hold_return = strategy1.forecast.close.pct_change()
    buy_hold_cum_return = (buy_hold_return + 1).cumprod()

    ffm_return = pd.Series(0.0, index=ffm_forecast.index)
    hybrid_return = pd.Series(0.0, index=hybrid_forecast.index)
    transac_cost = 0

    for i, index in enumerate(ffm_test_forecast.index):
        if ffm_test_forecast.ix[index, "case_result"] == "Obvious_LONG":
            ffm_return[index] = (strategy1.forecast.close[i+1] - transac_cost)/strategy1.forecast.close[i] - 1
        elif ffm_test_forecast.ix[index, "case_result"] == "Obvious_SHORT":
            ffm_return[index] = -((strategy1.forecast.close[i+1] + transac_cost) / strategy1.forecast.close[i] - 1)
    for i, index in enumerate(hybrid_test_forecast.index):
        if (hybrid_test_forecast.ix[index, "case_result"] == "Obvious_LONG" or
                        hybrid_test_forecast.ix[index, "case_result"] == "LONG"):
            hybrid_return[index] = (strategy1.forecast.close[i+1] - transac_cost)/strategy1.forecast.close[i] - 1
        elif (hybrid_test_forecast.ix[index, "case_result"] == "Obvious_SHORT" or
                      hybrid_test_forecast.ix[index, "case_result"] == "SHORT"):
            hybrid_return[index] = -((strategy1.forecast.close[i+1] + transac_cost) / strategy1.forecast.close[i] - 1)

    ffm_cum_return = (ffm_return + 1).cumprod()
    hybrid_cum_return = (hybrid_return + 1).cumprod()

    #print "Long days: %s" % sum(ffm_test_forecast.case_result == "Obvious_LONG")
    #print "Short days: %s" % sum(ffm_test_forecast.case_result == "Obvious_SHORT")

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(buy_hold_cum_return[1:])
    ax.plot(ffm_cum_return[1:])
    ax.plot(hybrid_cum_return[1:])
    ax.legend(["buy_and_hold", "futures_forecast_model", "hybrid (ffm and effm)"], loc="best")
    plt.xlabel("Date")
    plt.ylabel("Cumulative asset (assuming 1 at initiation)")
    plt.title("Performance of hybrid system vs. Buy-and-Hold (transac_cost=$" + str(transac_cost) + ")")
    fig.savefig("result/Hybrid_AI_System/hybrid_performance_cost_" + str(transac_cost) + ".png")
    plt.close(fig)
    print "\n(FM performance chart generated and saved!)\n"



if __name__ == "__main__":
    main()
