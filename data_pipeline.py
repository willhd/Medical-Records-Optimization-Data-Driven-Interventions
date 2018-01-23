import boto3
import pyspark
import subprocess
import os
import sys
import tempfile
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import sys
import pprint

# establish connection to database that contains chart information and s3_urls
conn = psycopg2.connect(dbname='charts_development',
                        user='willhd',
                        host='localhost',
                        password='dbpass'
                        )
s3 = start_session()


def start_session():
    '''starts a bloom records s3 session'''
    session = boto3.Session(profile_name='bloom')
    s3 = session.client('s3')
    return s3


def get_chart_object(s3_client, Bucket, Key):
    """returns chart object from specified bucket and key"""
    chart_object = s3_client.get_object(Bucket=Bucket,
                                        Key=Key)
    return chart_object


def read_chart_from_object(chart_object):
    chart_txt = chart_object['Body'].read()
    return chart_txt


def chart_search(q):
    """queries postgress database and print results"""
    cur = conn.cursor()
    cur.execute(q)
    result = cur.fetchall()
    return result


def get_documents_ref(sql_table):
    """returns a list of lists of chart notes without duplicates that are pdf only, and where text ref is available"""
    return [txt_url if txt_url else pdf_url for (txt_url, pdf_url) in sql_table]


def read_chart_from_object(chart_object):
    chart_txt = chart_object['Body'].read()
    return chart_txt


def documents_to_pandas(s3_client, s3_urls, dataframe, coloumn):
    """returns pandas dataframe with encounter_note column from s3_url"""
    start_session()
    charts = []

    for s3_url in s3_urls:
        offset = len('s3://chartpull-agent-storage/')
        Key = s3_url[offset:]
        Bucket = 'chartpull-agent-storage'
        chart_note = get_chart_object(s3_client=s3_client,
                                      Bucket=Bucket,
                                      Key=Key)
        charts.append(chart_note)
    dataframe[column] = charts
    return dataframe


def df_col_to_text_object(df, col):
    text_objects = []
    for i in range(1, len(df)):
        text_object = df[col][i]
        text_objects.append(text_object['Body'].read())
    return text_objects

# file conversion


def s3totempfile(s3_client, s3_url):
    """download file from s3 and return tempfile name"""
    bucket = 'chartpull-agent-storage'
    offset = len('s3://chartpull-agent-storage/')
    key = s3_url[offset:]
    print(bucket, key)
    note = s3_client.get_object(Bucket=bucket, Key=key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.file.write(note['Body'].read())
        return tmp_file.name
