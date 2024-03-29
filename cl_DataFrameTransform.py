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

    def perc_of_loss_of_loans(self):
        # v_perc_of_loss = self.df[(self.df.loan_status == 'Charged Off')].count()
        v_perc_of_loss = (self.df[(self.df.loan_status == 'Charged Off')].shape[0] / self.df.shape[0]) * 100
        print('% of loan_status == Charged Off', v_perc_of_loss)

    def total_chared_off_amt(self):
        tot_chrg_off_loan = self.df[(self.df.loan_status == 'Charged Off')]['loan_amount'].sum()
        print('total loan_amount for Charged-off loans :',tot_chrg_off_loan) 
        tot_chrg_off_recory = self.df[(self.df.loan_status == 'Charged Off')]['recoveries'].sum()
        print('total recoveries for Charged-off loans :',tot_chrg_off_recory) 
        print('% of recoveries and loan_amount for Charged-off loans :', (tot_chrg_off_recory / tot_chrg_off_loan)*100)
    
    def loss_in_revenue_amt(self):
        ## Creating temp DF
        loans_subset_df = self.df[(self.df.loan_status == 'Charged Off')][['loan_amount', 'term', 'int_rate', 'recoveries']].copy()
        ## created term as numeric and appended to temp-df
        v_term_in_num = loans_subset_df['term'].str.slice(0,2).astype(float).fillna(0) 
        loans_subset_df['term_in_num'] = v_term_in_num
        ## calculated potential loss amt for charged-off loans and appended to temp-df
        v_tot_pot_amt = loans_subset_df['loan_amount'] * (loans_subset_df['term_in_num']/12) * loans_subset_df['int_rate'] 
        loans_subset_df['tot_pot_amt'] = v_tot_pot_amt
        ## deducted recoveries from potential loss amt for charged-off loans and appended to temp-df
        v_pot_amt_aftr_recry = loans_subset_df['tot_pot_amt'] - loans_subset_df['recoveries']
        loans_subset_df['pot_amt_aftr_recry'] = v_pot_amt_aftr_recry
        print(loans_subset_df.head())
        ## showing total potential loss amt for charged-off loans 
        print('Total loss in revenue for Charged-off loans :',loans_subset_df['pot_amt_aftr_recry'].sum()) 

    def loss_for_paymnts_delay(self):
        ## count of loans with status of Late
        v_paymnts_delay_counts = self.df[self.df['loan_status'].str.contains('Late')]['id'].shape[0]
        print('total v_paymnts_delay_counts for delayed payments :',v_paymnts_delay_counts) 
        ## Totoal count of loans
        v_tot_loans_count = self.df['id'].count()
        print('total tot_loans_count for delayed payments :',v_tot_loans_count) 
        ## percentage of delayed loans over total loans issued 
        print('% of delayed payments over total loans :', (v_paymnts_delay_counts / v_tot_loans_count)*100)

    def indicators_of_loss_by_grd_purp(self):
        ## method to show Counts of Purpose of Loan per allocated Grade
        loans_subset_df = self.df[['grade','purpose', 'home_ownership', 'loan_status']].copy()
        print(loans_subset_df.head())
        loans_subset_df = loans_subset_df.sort_values(by=["grade"])
        fig = px.histogram(loans_subset_df, "purpose", facet_col="grade",
             color="grade",
             title="Counts of Purpose of Loan per allocated Grade",
             labels={"grade": "Grade", "purpose": "Purpose"},
             height=2000,
             facet_col_wrap=2,
             facet_col_spacing=0.1)

        fig.update_layout(showlegend=False)
        fig.update_xaxes(showticklabels=True, tickangle=45)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.show()

    def indicators_of_loss_by_grd_sts(self):
        ## method to show Counts of Loan Status per allocated Grade
        loans_subset_df = self.df[['grade','purpose', 'home_ownership', 'loan_status']].copy()
        print(loans_subset_df.head())
        loans_subset_df = loans_subset_df.sort_values(by=["grade"])
        fig = px.histogram(loans_subset_df, "loan_status", facet_col="grade",
             color="grade",
             title="Counts of Loan Status per allocated Grade",
             labels={"grade": "Grade", "loan_status": "Loan Status"},
             height=2000,
             facet_col_wrap=2,
             facet_col_spacing=0.1)

        fig.update_layout(showlegend=False)
        fig.update_xaxes(showticklabels=True, tickangle=45)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.show()
