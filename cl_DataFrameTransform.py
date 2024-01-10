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

class DataFrameTransform:
    def load_data(self):
        # df.to_csv('loan_payments.csv', sep='\t')
        self.df = pd.read_csv("loan_payments.csv",index_col=0)
        # print('check data', self.df)

    def analyse_missing_values(self):
        print('isna info :', self.df.isna())
        print("percentage of missing values in each column:")
        print(self.df.isna().mean() * 100)

    def drop_col(self, col_name):
        # couldn't determine what and how to drop a column 
        self.df.dropna(axis=1, how='all')

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
        t=sns.histplot(self.df['loan_amount'],label="Skewness: %.12f"%(self.df['loan_amount'].skew()),kde=True )
        t.legend()
        qq_plot = qqplot(self.df['loan_amount'] , scale=1 ,line='q', fit=True)
        plt.show()
        self.df['loan_amount'].describe()
        # backing-up the dataframe
        # back-up_df=self.df
        # not sure how to take a back-up of a df as the above seem doesn't work
        print('After log transform')
        log_loan_amount = self.df["loan_amount"].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_loan_amount,label="Skewness: %.2f"%(log_loan_amount.skew()),kde=True )
        t.legend()
        qq_plot = qqplot(log_loan_amount, scale=1 ,line='q', fit=True)
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

    def per_of_loans_recovered(self):
        loans_subset_df = self.df[['funded_amount','total_payment', 'term', 'instalment']].copy()
        print(loans_subset_df.head())
        # loans_subset_df['perc_of_ln_recd'] = self.df(['total_payment'] / ['funded_amount'])
        # loans_subset_df = loans_subset_df.assign(perc_of_ln_recd=self.df(['total_payment'] / ['funded_amount']))
        # loans_subset_df['perc_of_ln_recd'] = self.df['total_payment'] / self.df['funded_amount']
        v_term_in_num = self.df['term'].str.slice(0,2).astype(float).fillna(1) 
        loans_subset_df['term_in_num'] = v_term_in_num
        # print(loans_subset_df.head(20))
        v_actual_tot_amt = loans_subset_df['term_in_num'] * loans_subset_df['instalment'].astype(float).fillna(1)
        loans_subset_df['actual_tot_amt'] = v_actual_tot_amt 
        v_perc_of_ln_recd = loans_subset_df['total_payment'] / loans_subset_df['actual_tot_amt']*100 
        loans_subset_df['perc_of_ln_recd'] = v_perc_of_ln_recd 
        print(loans_subset_df.head())
        sns.scatterplot(y=loans_subset_df['funded_amount'], x=loans_subset_df['perc_of_ln_recd'])
        plt.show()

