## all imports of packages
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import pylab
from scipy.stats import normaltest
import scipy.stats as stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

class DataCleaning:
    ## Class initializations 
    def __init__(self, df):
        ## Variables of the class being initialized 
        self.df = df
    
    ## function definition
    def load_data(self):
        # df.to_csv('loan_payments.csv', sep='\t')
        self.df = pd.read_csv("loan_payments.csv",index_col=0)
        # print('check data', self.df)

    def analyse_nulls(self):
        print('percentage of null values in each column:')
        null_stats = self.df.isnull().sum()/len(self.df)
        for line in null_stats:
            print(line)

    def analyse_missing_values(self):
        # print('isna info :', self.df.isna())
        print("percentage of missing values in each column:")
        print(self.df.isna().mean() * 100)

    def drop_col(self, col_name):
        # couldn't determine what and how to drop a column 
        self.df.dropna(axis=1, how='all')

    def drop_col_with_nulls(self):
        missing_percentage = self.df.isna().mean() * 100
        threshhold = 80
        cleaned_df = self.df.drop(columns=missing_percentage[missing_percentage > threshhold].index)

    def impute_col(self):
        # imputed_column = self.df['mths_since_last_delinq'].fillna(self.df['mths_since_last_delinq'].median())  # Median
        self.df['mths_since_last_delinq']=self.df['mths_since_last_delinq'].fillna(self.df['mths_since_last_delinq'].median())  # Median
        msno.matrix(self.df)
        plt.show()

    def find_skewed_columns(self):
        numeric_features = self.df.select_dtypes(include='number')
        categorical_features = [col for col in self.df.columns if col not in numeric_features]
        sns.set(font_scale=0.7)
        f = pd.melt(self.df, value_vars=numeric_features)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        print(categorical_features)


    def fix_skew_log_transform(self):
        print('Before log transform')
        t=sns.histplot(self.df['loan_amount'],label="Skewness: %.2f"%(self.df['loan_amount'].skew()),kde=True )
        t.legend()
        plt.show()
        # qq_plot = qqplot(self.df['loan_amount'] , scale=1 ,line='q', fit=True)
        # plt.show()
        # self.df['loan_amount'].describe()
        # backing-up the dataframe
        # back-up_df=self.df
        # not sure how to take a back-up of a df as the above seem doesn't work
        print('loan_amount - After Log Transform')
        log_loan_amount = self.df["loan_amount"].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_loan_amount,label="Skewness: %.2f"%(log_loan_amount.skew()),kde=True )
        t.legend()
        # qq_plot = qqplot(log_loan_amount, scale=1 ,line='q', fit=True)
        plt.show()
     
        print('loan_amount - After Box-cox Transformation')
        boxcox_population = self.df["loan_amount"]
        boxcox_population= stats.boxcox(boxcox_population)
        boxcox_population= pd.Series(boxcox_population[0])
        t=sns.histplot(boxcox_population,label="Skewness: %.2f"%(boxcox_population.skew()) )
        t.legend()
        # qq_plot = qqplot(log_loan_amount, scale=1 ,line='q', fit=True)
        plt.show()

        print('loan_amount - After Yeo-Johnson Transformation')
        yeojohnson_population = self.df["loan_amount"]
        yeojohnson_population = stats.yeojohnson(yeojohnson_population)
        yeojohnson_population= pd.Series(yeojohnson_population[0])
        t=sns.histplot(yeojohnson_population,label="Skewness: %.2f"%(yeojohnson_population.skew()) )
        t.legend()
        plt.show()

        # Checking skewness on another variable
        print('total_payment - Log Transform')
        log_transform_population = self.df["total_payment"].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_transform_population,label="Skewness: %.2f"%(log_transform_population.skew()),kde=True )
        t.legend()
        # qq_plot = qqplot(log_loan_amount, scale=1 ,line='q', fit=True)
        plt.show()
     
        print('total_payment - Box-cox Transformation')
        boxcox_population = self.df["total_payment"]
        boxcox_population= stats.boxcox(boxcox_population)
        boxcox_population= pd.Series(boxcox_population[0])
        t=sns.histplot(boxcox_population,label="Skewness: %.2f"%(boxcox_population.skew()) )
        t.legend()
        # qq_plot = qqplot(log_loan_amount, scale=1 ,line='q', fit=True)
        plt.show()

        print('total_payment - Yeo-Johnson Transformation')
        yeojohnson_population = self.df["total_payment"]
        yeojohnson_population = stats.yeojohnson(yeojohnson_population)
        yeojohnson_population = pd.Series(yeojohnson_population[0])
        t=sns.histplot(yeojohnson_population,label="Skewness: %.2f"%(yeojohnson_population.skew()) )
        t.legend()
        plt.show()

    def remove_outliers_zscore(self):
        self.df[(np.abs(stats.zscore(self.df)) < 3).all(axis=1)]
        sns.boxplot(data=self.df)
        plt.show()

    def remove_outliers_using_df_col_criteria(self):
        # Plotting scatter plots for both datasets
        print('Removing outliers found using Scatter plot')
        self.df = self.df.drop(self.df[(self.df.int_rate == 6) & (self.df.grade.isin(['B','C','D','E','F']))].index)
        sns.scatterplot(y=self.df['int_rate'], x=self.df['grade'])
        plt.show()


