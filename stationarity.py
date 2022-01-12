from statsmodels.tsa.stattools import adfuller as adf
from pandas import read_csv, DataFrame, to_datetime
from matplotlib import pyplot as plt


class Stationarity_diff:
    """
    Study of data stationarity.
    
    Required settings:
        - data (dependent variable)
        - variable (formatted dependent variable - "NAME VARIABLE")
        - p_value_accepted (p-value number accepted)
        
    Syntax: Stationarity_diff(data_endog, data_exogs, variable name, p-value)
    """

    def __init__(self, data, variable, p_value_accepted=0.05):
        """
        Settings for the outputs.
        """
        
        # data frame
        self.data_endog = data.iloc[ : , 0 : 1 ]
        self.data_exogs = data.iloc[ : , 1 :   ]
        self.data_all = data
        
        # configs
        self.variable = variable
        self.variable_ = variable.replace(" ", "_").lower()
        self.p_value_accepted = p_value_accepted


    def adf_teste(self):
        """
        Adf test.
        """
        
        adf_level = adf(self.data_endog, regression='ct')
        
        adf_level_result = (
        f"{'-' * 50}\n"
        f"ADF Results (level):\n\n"
        f"Variable: {self.variable}\n"
        f"ADF Test: {adf_level[0]:.6f}\n"
        f"P-value: {adf_level[1]:.6f}\n"
        f"Lags: {adf_level[2]}\n"
        f"Observations: {adf_level[3]}\n"
        f"Critical values:\n"
        f"  1%: {adf_level[4]['1%']:.6f}\n"
        f"  5%: {adf_level[4]['5%']:.6f}\n"
        f"  10%: {adf_level[4]['10%']:.6f}\n"
        f"{'-' * 50}")
        
        with open('4_results/7_adf_test_level.txt', 'w') as desc_stat:
            desc_stat.write(adf_level_result)
        
        return


    def diff_data(self):
        """"
        Function that returns the stationary series through the ADF test criterion 
        and differentiation method. The function will also set the value of 
        parameter (d).
        """
        
        count_diff = 0
        
        while True:
            adf_test_diff = adf(self.data_endog, regression='ct')
            adf_p_value = adf_test_diff[1]
            
            if adf_p_value > self.p_value_accepted:
                stationary_series = self.data_endog.diff().dropna()
                self.data_endog = stationary_series
                count_diff += 1
            
            else:
                adf_diff = adf(self.data_endog, regression='ct')
                
                adf_result = (
                f"{'-' * 50}\n"
                f"ADF Results:\n\n"
                f"Nonseasonal differences needed for stationarity: {count_diff}\n\n"
                f"Variable: {self.variable}\n"
                f"ADF Test: {adf_diff[0]:.6f}\n"
                f"P-value: {adf_diff[1]:.6f}\n"
                f"Lags: {adf_diff[2]}\n"
                f"Observations: {adf_diff[3]}\n"
                f"Critical values:\n"
                f"  1%: {adf_diff[4]['1%']:.6f}\n"
                f"  5%: {adf_diff[4]['5%']:.6f}\n"
                f"  10%: {adf_diff[4]['10%']:.6f}\n"
                f"{'-' * 50}")
                
                with open('4_results/8_adf_diff_result.txt', 'w') as desc_stat:
                    desc_stat.write(adf_result)
                
                break
        
        self.data_endog.to_csv("3_working/2.0_stationary_endog.csv")
        
        return


    def independent_var_stationarity(self):
        """
        Treatment of stationarity of independent variables.
        """
        
        list_exog_col = self.data_exogs.columns.to_list()
        
        for col in list_exog_col:
            while True:
                adf_test_diff = adf(self.data_exogs[col], regression='ct')
                adf_p_value = adf_test_diff[1]
                
                if adf_p_value > self.p_value_accepted:
                    stat_col = self.data_exogs[col].diff().dropna()
                    self.data_exogs[col] = stat_col
                
                else:
                    break
        
        self.data_exogs.to_csv("3_working/2.1_stationary_exog.csv")
        
        return

