import config
import warnings
from pandas import read_csv, to_datetime, concat
from descriptive_statistics import Time_serie_level
from x13arima_seas_adjust import X13_arima_desaz
from stationarity import Stationarity_diff
from model_execute import Model_execute
from data_input import variable, data_endog, data_exogs


# suppress warnings - sorry about that =(
warnings.filterwarnings("ignore")


# Time_serie_level (descriptive statistics) {
# ==========================================================================

descriptive_statistics = Time_serie_level(data_endog, variable,
			 config.style_graph, config.color1, config.color2, 
                         config.color3, config.color4, config.color5)

descriptive_statistics.time_serie_plot()
descriptive_statistics.acf_pacf_plot()
descriptive_statistics.periodogram_plot()
descriptive_statistics.descriptive_stat()

# ==========================================================================
# }


# seasonality {
# ==========================================================================

# x13-arima-seats {
# --------------------------------------------------------------------------
x13_desaz = X13_arima_desaz(data_endog, 
                            data_exogs,
                            variable,
                            config.path_x13_arima,
                            config.style_graph, config.color1, config.color2, 
                            config.color3, config.color4, config.color5)

x13_desaz.x13_results()
x13_desaz.x13_seasonal_adjustment()
x13_desaz.independent_desaz_x13()
# --------------------------------------------------------------------------
#   }

# ==========================================================================
# }


# stationarity {
# ==========================================================================

try:
    data_non_seasonal = read_csv("3_working/1_seasonal_adjustment.csv", sep=",", decimal=".")
    data_non_seasonal["index_date"] = to_datetime(data_non_seasonal["index_date"])
    data_non_seasonal = data_non_seasonal.sort_values("index_date")
    data_non_seasonal = data_non_seasonal.set_index("index_date")

except:
    print(f"\n\n\n{'=' * 80}\n\n")
    print(">>> The file with non-seasonal data '1_seasonal_adjustment.csv' does not exist in the folder '3_working'. You have the options:\n\n")
    print("• Copy, paste and rename the file 'data_base.csv' to '1_seasonal_adjustment.csv' and work with the data without seasonality treatment;\n")
    print("• Activate X13-ARIMA-SEATS in the 'main.py' file;\n")
    print("• Use other methods, such as: decomposition, Hodrick–Prescott filter or seasonal dummies (don't forget to put the file in the '3_working' folder and rename the file to '1_seasonal_adjustment.csv'.")
    print(f"\n\n{'=' * 80}\n\n")
    exit()

stationarity = Stationarity_diff(data_non_seasonal, variable, config.p_value_accepted)

stationarity.adf_teste()
stationarity.diff_data()
stationarity.independent_var_stationarity()

# ==========================================================================
# }


# model execute {
# ==========================================================================

# open stationary data
# --------------------------------------------------------------------------

try:
    data_stationarity = read_csv("3_working/2.1_stationary_exog.csv", sep=",", decimal=".")

    data_stationarity["index_date"] = to_datetime(data_stationarity["index_date"])
    data_stationarity = data_stationarity.sort_values("index_date")
    data_stationarity = data_stationarity.set_index("index_date")

except:
    print(f"\n\n\n{'=' * 80}\n\n")
    print("The file '2.1_stationary_exog.csv' was not found in the '3_working' folder.")
    print(f"\n\n{'=' * 80}\n\n")
    exit()

# --------------------------------------------------------------------------
#   }

# model execute {
# --------------------------------------------------------------------------
data_non_seasonal_endog = read_csv("3_working/1_seasonal_adjustment.csv", 
                                   sep=",", decimal=".")

data_non_seasonal_endog["index_date"] = to_datetime(data_non_seasonal_endog["index_date"])
data_non_seasonal_endog = data_non_seasonal_endog.sort_values("index_date")
data_non_seasonal_endog = data_non_seasonal_endog.set_index("index_date")

data_model = concat([data_non_seasonal_endog,
                     data_stationarity.iloc[ : , 2 : ]], 
                     axis=1)

model = Model_execute(data_model, variable,
                      config.style_graph, config.color1, config.color2, 
                      config.color3, config.color4, config.color5)

model.model_execute(config.p, config.d, config.q, 
                    config.P, config.D, config.Q, config.s)

model.residuals_analysis()
model.acf_pacf_residuals()
model.ts_residuals_plot()
model.adjust_predict()
# --------------------------------------------------------------------------
#   }

# ==========================================================================
# }

