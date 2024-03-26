# bending test
import numpy as np
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
import statsmodels.api as sm

dot_num = 40
cycling = True

def cycle_data_align(datafile, cycle_number):
    # cycle_number = [strain rate number, cycle number at each strain], e.g, [3, 5]
    strain_number = cycle_number[0]  # mm, the width of the tested sample
    cn_strain = cycle_number[1]  # um, the thickness of the tested sample

    data_file = pd.read_csv(datafile)
    index_list = data_file[data_file["Project:"] == "Interval and data points:"].index.to_numpy()

    interval_data = []
    for i in range(len(index_list)-1):
        if i < len(index_list)-2:
            interval_data_i = data_file[index_list[i+1]:index_list[i+2]][:]
        else:
            interval_data_i = data_file[index_list[i+1]:][:]
        interval_data_i = interval_data_i[1:][:]
        header_row = interval_data_i.iloc[0]
        interval_data_i = pd.DataFrame(interval_data_i.values[1:], columns=header_row)
        clean_interval_i = interval_data_i[["Extensional Strain","Extensional Stress"]]
        # drop the rows without numerical data
        clean_interval_i = clean_interval_i[2:][:-2]
        interval_data.append(clean_interval_i)

    all_data = pd.DataFrame()
    dict = {}
    j = 0
    while j < len(interval_data)-1:
        cycle_index = int(j/2+1)
        print("cycle at {}".format(cycle_index))
        interval_combine_j = interval_data[j].append(interval_data[j+1])
        interval_combine_j = interval_combine_j.reset_index(drop=True)
        dict["Cycle at " + str(cycle_index)] = interval_combine_j
        all_data = pd.concat([all_data,interval_combine_j], axis=1)
        j = j + 2
    #print(dict)

    if cycling: # long cycle at a specific strain limit and strain rate.
        df_pick = pd.DataFrame()
        for s in range(0, 301, 30):

            picked_columns = all_data.iloc[:, s:(s+2)].reset_index(drop=True)
            df_pick = pd.concat([df_pick, picked_columns], axis=1)

        df_pick.to_csv("picked_data.csv")

    else: # rate cycling with different strain limit and strain rates
        all_data.to_csv("all_data.csv")
    return dict



def linear_fit(dict):
    cycle_idx_set = []
    data_set = []
    new_dict = {}
    for k, p in dict.items():

        for l, s in p.items():
            cycle_idx_set.append(l)
            data_set.append(s)

        slope_set = []
        for t in range(len(cycle_idx_set)):
            x_raw = np.array(data_set[t]['Extensional Strain'][:230]).astype(float)
            y_raw = np.array(data_set[t]['Extensional Stress'][:230]).astype(float)
            yhat = savgol_filter(y_raw, 51, 3)
            x = np.array([m for (m, n) in zip(x_raw, yhat) if m > 0 and n > 0][:dot_num])
            y = np.array([n for (m, n) in zip(x_raw, yhat) if m > 0 and n > 0][:dot_num])

            x = x.reshape(-1, 1)
            regression_model = LinearRegression()

            # Fit the data(train the model)
            regression_model.fit(x, y)

            # Predict
            y_predicted = regression_model.predict(x)

            # model evaluation
            mse = mean_squared_error(y, y_predicted)

            rmse = np.sqrt(mean_squared_error(y, y_predicted))
            r2 = r2_score(y, y_predicted)

            slope_b1 = regression_model.coef_
            intercept_b0 = regression_model.intercept_
            slope_set.append(slope_b1[0])

            # printing values
            """
            print('Slope:', regression_model.coef_)
            print('Intercept:', regression_model.intercept_)
            print('MSE:', mse)
            print('Root mean squared error: ', rmse)
            """
            print('Cycle: ', t)
            print('R2 score: ', r2)


            y_pred = slope_b1 * x + intercept_b0

            fig, ax = plt.subplots()
            ax.set_xlabel('Extensional Strain')
            ax.set_ylabel('Extensional Stress (Pa)', color='tab:red')
            mpl.rc("font", family="Times New Roman", weight='normal')
            plt.rcParams.update({'mathtext.default': 'regular'})

            plt.scatter(x_raw, y_raw, color='red')
            plt.plot(x_raw, yhat, color='blue')
            plt.plot(x, y_pred, color='green')

            ax.set_title('linear fit from initial data points greater than 0')
            #fig.savefig(str(k) + str(cycle_idx_set[t])+'.png')
            plt.show()

        df = pd.DataFrame(list(zip(cycle_idx_set, slope_set)), columns=['Cycle', 'Fitting Slope'])
        new_dict[str(k)] = df
        df_out = pd.DataFrame.from_dict(df)
        df_out.to_csv(str(k) +" linear_fit.csv")
