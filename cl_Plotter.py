## all imports of packages
import pandas as pd
from scipy.stats import normaltest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import pylab
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px

class Plotter:
    def load_data(self):
        # df.to_csv('loan_payments.csv', sep='\t')
        self.df = pd.read_csv("loan_payments.csv",index_col=0)
        # print('check data', self.df)

    def analyse_missing_values(self):
        print('isna info :', self.df.isna())
        print("percentage of missing values in each column:")
        print(self.df.isna().mean() * 100)

    def identify_outliers(self):
        # First visualise your data using your Plotter class to determine if the columns contain outliers.
        print('Using BoxPlot to identify outliers')
        sns.boxplot(data=self.df)
        plt.show()
        # Plotting scatter plots for two columns 
        print('Using Scatter plot to identify outliers')
        sns.scatterplot(y=self.df['int_rate'], x=self.df['grade'])
        plt.show()

    def identify_Collinearity(self):
        loans_subset_df = self.df[['loan_amount', 'int_rate' ]]
        # loans_subset_df = self.df.astype[['int_rate': float, 'grade']]
        px.imshow(loans_subset_df.corr(), title="Correlation heatmap of loans dataframe")
        plt.show()

