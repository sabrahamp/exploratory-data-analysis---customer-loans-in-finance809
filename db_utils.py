## all imports of packages
import yaml
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
from sqlalchemy import text

class RDSDatabaseConnector:
    def __init__(self):
        print('Loading YAML file ....')
        def load_yaml():
            with open('credentials.yaml', 'r') as file:
                return yaml.safe_load(file)
        ## Load connection credentials from yaml file
        self.cred= load_yaml()

        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'

    def db_connect(self):
        # Establish the connection
        print('Establishing the connection....')
        self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.cred['RDS_USER']}:{self.cred['RDS_PASSWORD']}@{self.cred['RDS_HOST']}:{self.cred['RDS_PORT']}/{self.cred['RDS_DATABASE']}")

    def db_extract_data(self):
        print('Extracting the data to Dataframe and write to a csv file...')
        with self.engine.execution_options(isolation_level='AUTOCOMMIT').connect() as conn:
            # table named 'loan_payments' will be returned as a dataframe.
            df = pd.read_sql_table('loan_payments', conn)
            df.to_csv('loan_payments.csv', sep='\t')

con_db = RDSDatabaseConnector()
con_db.db_connect()
con_db.db_extract_data()
