## all imports of packages
import pandas as pd
from scipy.stats import normaltest

class DataTransform :
    def load_data(self):
        # df.to_csv('loan_payments.csv', sep='\t')
        self.df = pd.read_csv("./loan_payments.csv",index_col=0)
        print(self.df)
        print(self.df.columns())
        print(self.df.index())
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


dt = DataTransform()
dt.load_data()
dt.analyse_nulls()
# dt.is_normally_dist()
# dt.con_dtypes()
