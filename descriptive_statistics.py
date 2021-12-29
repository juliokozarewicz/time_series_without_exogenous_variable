from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from numpy import absolute as abs
from numpy.fft import fft
from numpy import std
from numpy import var


class Time_serie_level:
    """
    Elaboration of descriptive statistics results.

    Required settings:
    - data (input_data)
    - variable (formatted dependent variable - "NAME VARIABLE")

    Optional settings:
    - style_graph (graphic style)
    - color1 (color setting)
    - color2 (color setting)
    - color3 (color setting)
    - color4 (color setting)
    - color5 (color setting)
    
    Syntax: Time_serie_level(data, variable, style_graph,
                                             color1, color2, color3, color4, color5)

    """

    def __init__(self, data, variable,
                 style_graph="seaborn", 
                 color1="royalblue", 
                 color2="crimson", 
                 color3="darkorange", 
                 color4="black", 
                 color5="red"):
        """
        Settings for the outputs.
        """
        
        # data frame
        self.data_select = data
        
        # configs
        self.variable = variable
        self.variable_ = variable.replace(" ", "_").lower()
        
        # style
        self.style_graph = style_graph
        self.color1 = color1
        self.color2 = color2
        self.color3 = color3
        self.color4 = color4
        self.color5 = color5


    def time_serie_plot(self):
        """
        Time series plot.
        """
        
        # style
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        # set
        plt.title(f"TIME SERIE (LEVEL) - {self.variable}")
        
        # plot
        ts_plot = plt.plot(self.data_select, linestyle="solid",
                                        color=self.color1, 
                                        linewidth = 2)
        
        # format
        plt.gcf().autofmt_xdate() # year
        date_format = mpl_dates.DateFormatter('%b. %Y') # month, year
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.tight_layout()
        
        # save
        plt.savefig(f"4_results/1_time_serie.jpg")
        
        return


    def acf_pacf_plot(self):
        """
        ACF and PACF plots.
        """
        
        # style
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6), dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        # plot
        acf = plot_acf(self.data_select.values.squeeze(),
                       lags = len(self.data_select) / 3,
                       use_vlines = True,
                       title = f"ACF (LEVEL) - {self.variable}",
                       color = self.color1,
                       vlines_kwargs = {"colors": self.color1},
                       alpha=0.05,
                       ax=ax[0],
                       zero=True)
        
        acf = plot_pacf(self.data_select.values.squeeze(),
                       lags = len(self.data_select) / 3,
                       use_vlines = True,
                       title = f"PACF (LEVEL) - {self.variable}",
                       color = self.color2,
                       vlines_kwargs = {"colors": self.color2},
                       alpha=0.05,
                       ax=ax[1],
                       zero=True)
        
        ax[0].set_ylim(-0.8, 1.1) 
        ax[1].set_ylim(-0.7, 1.1)
        plt.tight_layout()
        
        # save
        fig.savefig(f"4_results/2_fac_facp_level.jpg")
        
        return


    def periodogram_plot(self):
        """
        Periodogram plot.
        """

        # selection
        self.data_select = self.data_select.iloc[ : , 0 ]
        
        # style
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 6), dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        # plot
        ps = abs(fft(self.data_select, n=len(self.data_select / 4 )  ) )  ** 2
        periodogram = plt.plot(ps, color=self.color1)
        
        # save
        plt.title(f"PERIODOGRAM - {self.variable} (LEVEL)")
        plt.tight_layout()
        fig.savefig(f"4_results/3_periodogram_level.jpg")
        
        return


    def descriptive_stat(self):
        """
        Descriptive data analysis
        """
       
        # define variables
        mean = self.data_select.mean()
        median = self.data_select.median()
        std_sample = std(self.data_select)
        variance = var(self.data_select)
        lowest = self.data_select.min()
        highest = self.data_select.max()
        
        # frame
        results_txt = (
        f"{'-' * 50}\n"
        f"Descriptive analysis:\n\n"
        f"Variable: {self.variable} (level)\n"
        f"Mean: {mean:.2f}\n"
        f"Median: {median:.2f}\n"
        f"Sample std: {std_sample:.2f}\n"
        f"Variance: {variance:.2f}\n"
        f"Lowest: {lowest:.2f}\n"
        f"Highest: {highest:.2f}\n"
        f"{'-' * 50}\n"
        )
        
        # export
        with open('4_results/4_level_descriptive_statistics.txt', 'w') as desc_stat:
            desc_stat.write(results_txt)

        return
