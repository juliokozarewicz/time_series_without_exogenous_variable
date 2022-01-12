from pandas import DataFrame
from pandas import to_datetime
from pandas import read_csv


# data input
# ==========================================================
data_entry = read_csv("3_working/data_base.csv", sep=",", decimal=".")
data_entry["index_date"] = to_datetime(data_entry["index_date"])
data_select = data_entry.sort_values("index_date")

# date filter
data_select = data_select[ (data_select.iloc[:,0] >= '2000-01') ]

# variable for executing the model estimation
data_select = data_select.set_index("index_date")
# ==========================================================

# seasonally adjusted data entry
# ==========================================================
try:
    data_non_seasonal = read_csv("3_working/seasonal_adjustment.csv",
                                     sep=",", decimal=".")
    data_non_seasonal["index_date"] = to_datetime(data_non_seasonal["index_date"])
    data_non_seasonal = data_non_seasonal.sort_values("index_date")

    # variable for executing the model estimation
    data_non_seasonal = data_non_seasonal.set_index("index_date")

except:
    data_non_seasonal = data_select
# ==========================================================

# variables
# ==========================================================
variable_ = list(data_select.columns.values.tolist())[0]
variable = variable_.replace("_", ' ').upper()
p_value_accepted = 0.05

# X13 path linux
#path_x13_arima = "/home/user/x13as/"

# X13 path win
path_x13_arima = "C:/Program Files (x86)/x13as/0_x13as"
# ==========================================================

# Manual parameter setting
# ==========================================================
p = 1 #1
d = 1 #1
q = 1 #1
P = 3 #3
D = 1 #1
Q = 0 #0

# inform the periodicity of the series (D=365, M=12, Y=1)
s = 12
# ==========================================================

# style
# ==========================================================
style_graph = "seaborn"
color1 = "royalblue"
color2 = "indigo"
color3 = "darkorange"
color4 = "black"
color5 = "red"
# ==========================================================
