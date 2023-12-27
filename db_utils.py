import yaml
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
from sqlalchemy import text

cred_data=[]
v_cred_datai=[]

class RDSDatabaseConnector:
    def __init__(self, cred_data):
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'
        self.ENDPOINT = cred_data['RDS_HOST'] # "eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com"
        self.USER = cred_data['RDS_USER'] # 'loansanalyst'
        self.PASSWORD = cred_data['RDS_PASSWORD'] # "EDAloananalyst"
        self.PORT = cred_data['RDS_PORT'] # 5432
        self.DATABASE = cred_data['RDS_DATABASE'] # 'payments'

    def db_connect():
        # Establish the connection
        engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.ENDPOINT}:{self.PORT}/{self.DATABASE}")

    def db_extract_data():
        with engine.execution_options(isolation_level='AUTOCOMMIT').connect() as conn:
            # table named 'loan_payments' will be returned as a dataframe.
            df = pd.read_sql_table('loan_payments', conn)
            print(df)
        #df.head()


def load_yaml():
    with open('credentials.yaml', 'r') as file:
        cred_data = yaml.safe_load(file)

    print(f"The dictionary representation of the YAML file is: {cred_data}")
    return cred_data

# print('v_cred_data', v_cred_data)
# print(v_cred_data['RDS_HOST'])

v_cred_data = load_yaml()
con_db = RDSDatabaseConnector(v_cred_data)
con_db.db_extract_data()


# DATABASE_TYPE = 'postgresql'
# DBAPI = 'psycopg2'
# ENDPOINT = "eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com"
# USER = 'loansanalyst'
# PASSWORD = "EDAloananalyst"
# PORT = 5432
# DATABASE = 'payments'
# engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")

# engine.connect()
# with engine.connect() as connection:

# with engine.execution_options(isolation_level='AUTOCOMMIT').connect() as conn:
    # table named 'loan_payments' will be returned as a dataframe.
 #    df = pd.read_sql_table('loan_payments', conn)
 #    print(df)
    #df.head()

#print(df)

#    data = conn.execute(text("SELECT * FROM loan_payments"))
#loan = pd.DataFrame(data['data'], columns=data['feature_names'])
#loan.head()
