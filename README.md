# exploratory-data-analysis---customer-loans-in-finance809

Project Title
Exploratory datax -analysis-customer-loans-in-finance809

Table of Contents, if the README file is long
    -   A description of the project: what it does, the aim of the project, and what you learned
    -   Installation instructions
    -   Usage instructions
    -   File structure of the project
    -   License information

A description of the project: what it does, the aim of the project, and what you learned
    To ensure informed decisions are made about loan approvals and risk is efficiently managed, this project is to gain a comprehensive understanding of the loan portfolio data.
I'm working on a Data-Analysis project for large financial institution, where managing loans is a critical component of business operations. 
    the main script called db_utils,py has a class which has methods that :
        -   reads a yaml file to get DB-credentials
        -   connects to AWS DB called payments
        -   extracts data from loan_payments table and writes to a ,csv file

Installation instructions
    Please follow standard Python installation and othere packages e.g. pandas, yaml, sqlalchemy etc.
Usage instructions
    To execute please run - Python db_utils.py
    Use main ipynb file to go through each task of a Milestone and do the analysis by executing the method
File structure of the project
ipvnb file calls the methods from the .py files
    RDSDatabaseConnector
        __init__
            load_yaml()
                file
        cred
        DATABASE_TYPE
        DBAPI
        db_connect()
        engine()
        db_extract_date()
            conn
            df
        con_db
License information
    Standard Licence
