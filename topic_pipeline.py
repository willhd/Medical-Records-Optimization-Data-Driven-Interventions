import boto3
import pandas as pd
import numpy as np
import psycopg2
import string

""" The purpose of this data pipeline is to secrurely to connect to a s3 bucket and
postgreSQL data base """


class DataCollection(object):
    def __init__(self, dbname, user, host, password, profile_name, s3_client, Bucket, suffix):
        """
        INPUT:
        - dbname = name of SQL database to connect to
        - user = user_name for SQL database
        - host = server for SQL connection, default=localhost
        - password = password to SQL database
        - profile_name= AWS s3 profile_name
        - s3_client = s3_client name
        - Key = AWS s3_bucker key
        - Bucket = name of s3_bucket
        - suffix = first component of s3_url ('s3://name-of-user-bucket/')

        ATTRIBUTES:
        - chart_search
        - sql_to_pandas
        - docs_to_pandas
        - save_data
        """
        self.dbname = dbname
        self.user = user
        self.host = host
        self.password = password
        self.profile_name = profile_name
        self.s3_client = s3_client
        self.Bucket = Bucket
        self.suffix = suffix
        self.conn = None
        self.dataframe = None
        self.s3 = None

    def connect_to_sql_database(self):
        """establish connection to database and returns connection"""
        conn = psycopg2.connect(dbname=self.dbname,
                                user=self.user,
                                host=self.host,
                                password=self.password
                                )
        self.conn = conn

    def chart_search(self, q):
        """queries postgress database and returns result"""
        cur = self.conn.cursor()
        cur.execute(q)
        result = cur.fetchall()
        self.sql_result = result

    def sql_to_pandas(self, columns):
        """returns pandas dataframe from sql querry result """
        dataframe = pd.DataFrame(columns=columns)
        for i in range(len(columns)):
            dataframe[columns[i]] = [item[i] for item in self.sql_result]
        self.dataframe = dataframe

    def start_session(self):
        ''' start an s3 session with boto3, returns s3 session object'''
        session = boto3.Session(profile_name=self.profile_name)
        s3 = session.client(self.s3_client)
        self.s3 = s3

    def get_chart_data(self, Key):
        """ open s3_bucket,
            get the data
            close the streaming text body
            return the text data """
        chart_object = self.s3.get_object(Bucket=self.Bucket, Key=Key)
        chart_body = chart_object['Body']
        chart_data = chart_body.read()
        chart_body.close()
        return chart_data

    def docs_to_pandas(self, column):
        """returns pandas dataframe with text column from s3_url
        """
        s3_urls = self.dataframe['s3_urls'].values
        texts = []
        for s3_url in s3_urls:
            offset = len(self.suffix)
            Key = s3_url[offset:]
            Bucket = self.Bucket
            raw_text = self.get_chart_data(Key=Key)
            texts.append(raw_text)
        self.dataframe[column] = texts

    def save_data(self):
        with open('topic_data.json', 'w') as f:
            # Write the model to a file as a JSON string .
            json.dump(self.dataframe, f)


if __name__ == '__main__':
    #topic_data = DataCollection()
