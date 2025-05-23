## all imports of packages
import pandas as pd
from scipy.stats import normaltest
from datetime import datetime
import numpy as np
import plotly.express as px

class DataAnalysis:
    ## Class initializations 
    def __init__(self, df):
        ## Variables of the class being initialized 
        self.df = df

    # def load_data(self):
        # df.to_csv('loan_payments.csv', sep='\t')
        self.df = pd.read_csv('loan_payments.csv',index_col=0)

    def analyse_data(self):
        print('printing data info :',self.df)
        print('Info :', self.df.info())
        print('Describe :',self.df.describe())
        # print(self.df.columns())
        # print(self.df.index())
        print(self.df["loan_amount"])

    def analyse_nulls(self):
        print('percentage of null values in each column:')
        null_stats = self.df.isnull().sum()/len(self.df)
        for line in null_stats:
            print(line)

    def is_normally_dist(self):
        data = self.df['loan_amount']
        print(data)
        # D’Agostino’s K^2 Test
        stat, p = normaltest(data, nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def con_dtypes(self):
        df=self.df.convert_dtypes()
        print(df)

    # def per_of_loans_recovered(self):
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

    def per_of_loans_recovered(self):
        loans_subset_df = self.df[['loan_amount','term', 'int_rate','issue_date']].copy()
        print(loans_subset_df.head())
        v_term_in_num = self.df['term'].str.slice(0,2).astype(float).fillna(1) 
        loans_subset_df['term_in_num'] = v_term_in_num
        v_perc_of_ln_recd = loans_subset_df['loan_amount'] * (loans_subset_df['int_rate'] / 100) * (loans_subset_df['term_in_num'] / 12)
        loans_subset_df['perc_of_ln_recd'] = v_perc_of_ln_recd 

        # v_date_diff_in_mnts = pd.to_datetime(self.df.issue_date).month - pd.to_datetime.today().month
        # print(v_date_diff_in_mnts)
        
        # covert issue_date to datetime
        loans_subset_df['v_issue_date'] = pd.to_datetime(self.df['issue_date'])
        print(loans_subset_df['v_issue_date'])
        # Current date
        v_current_date = pd.to_datetime(datetime.today())
        print(v_current_date)

        # Calculate the difference in months
        loans_subset_df['months_diff'] = (v_current_date.year - loans_subset_df['v_issue_date'].dt.year) * 12 + \
                                            (v_current_date.month - loans_subset_df['v_issue_date'].dt.month)
        
        # if loans_subset_df['months_diff'] > 0 and loans_subset_df['months_diff'] <= 6:
        #     loans_subset_df['6_months_int'] = (loans_subset_df['months_diff'] / 12) * loans_subset_df['int_rate'] * loans_subset_df['loan_amount']
        # else:
        #     loans_subset_df['6_months_int'] = (6 / 12) * loans_subset_df['int_rate'] * loans_subset_df['loan_amount']
        
        # We cannot use "if" statement on an entire pandas Series, which is not allowed. 
        # We're trying to compare a whole column (loans_subset_df['months_diff']) using scalar if, which only works for single Boolean values.
        # Instead, used np.where() or Boolean masking to apply conditions element-wise across the DataFrame.
        loans_subset_df['6_months_int'] = np.where(
            (loans_subset_df['months_diff'] > 0) & (loans_subset_df['months_diff'] <= 6),
            (loans_subset_df['months_diff'] / 12) * loans_subset_df['int_rate'] * loans_subset_df['loan_amount'],
            (6 / 12) * loans_subset_df['int_rate'] * loans_subset_df['loan_amount']
)
        print(loans_subset_df)

    def perc_of_loss_of_loans(self):
        # v_perc_of_loss = self.df[(self.df.loan_status == 'Charged Off')].count()
        v_perc_of_loss = (self.df[(self.df.loan_status == 'Charged Off')].shape[0] / self.df.shape[0]) * 100
        print('% of loan_status == Charged Off', v_perc_of_loss)

    def total_chared_off_amt(self):
        # conditional sum of a column from a Dataframe
        tot_chrg_off_loan = self.df[(self.df.loan_status == 'Charged Off')]['loan_amount'].sum()
        print('total loan_amount for Charged-off loans :',tot_chrg_off_loan) 
        tot_chrg_off_recory = self.df[(self.df.loan_status == 'Charged Off')]['recoveries'].sum()
        print('total recoveries for Charged-off loans :',tot_chrg_off_recory) 
        print('% of recoveries and loan_amount for Charged-off loans :', (tot_chrg_off_recory / tot_chrg_off_loan)*100)
        # isin function gives the same restricted result as above conditional sum from a Dataframe
        tot_chrg_off_loan = self.df[(self.df['loan_status'].isin(["Charged Off"]))]['loan_amount'].sum()
        print('total loan_amount for Charged-off loans using isin function:',tot_chrg_off_loan)
        # str.contains gives count of the string even if it is not present as a whole string but is part of a larger string
        tot_chrg_off_loan = self.df.loc[self.df['loan_status'].str.contains("Charged Off", na=False),'loan_amount'].sum()
        print('total loan_amount for Charged-off loans using str.contains function:',tot_chrg_off_loan) 
    
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

    def do_a_group_by(self):
        ## method to show Counts of Loan Status per allocated Grade
        loans_subset_df = self.df[['home_ownership']].copy()
        print(loans_subset_df.head())
        groupby_df = loans_subset_df.groupby(['home_ownership'])['home_ownership'].count()
        print(groupby_df)