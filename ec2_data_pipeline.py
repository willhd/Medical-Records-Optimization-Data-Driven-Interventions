import boto3
import pandas as pd
import numpy as np


def start_session(profile_name):
    '''starts s3 session with boto3'''
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client('s3')
    return s3


def get_chart_data(Bucket, Key):
    """ returns text from s3 """
    chart_object = s3.get_object(Bucket=Bucket, Key=Key)
    chart_body = chart_object['Body']
    chart_data = chart_body.read()
    chart_body.close()
    return chart_data


def docs_to_pandas(s3_client, s3_urls, dataframe):
    """returns pandas dataframe with encounter_note column from s3_url"""
    charts = []
    for s3_url in s3_urls:
        offset = len('s3://chartpull-agent-storage/')
        Key = s3_url[offset:]
        Bucket = 'chartpull-agent-storage'
        chart_note = get_chart_data(Bucket=Bucket, Key=Key)
        charts.append(chart_note)
    dataframe['encounter-note'] = charts
    return dataframe


def s3_links_to_text(s3_urls):
    """converts links to text body"""
    charts = []
    for s3_url in s3_urls:
        offset = len('s3://chartpull-agent-storage/')
        Key = s3_url[offset:]
        Bucket = 'chartpull-agent-storage'
        chart_note = get_chart_data(Bucket=Bucket, Key=Key)
        charts.append(chart_note)
    return charts


def read_data(path):
    """converts data to pandas dataframe"""
    docs_df = pd.read_csv()
    return docs_df


def get_urls(df):
    """ obtains urls as list from pandas dataframe"""
    s3_urls = list(docs_df['s3_url'])
    return s3_urls


if __name__ == "__main__":
    s3 = start_session()
    docs_df = read_data()
    s3_urls = get_urls(docs_df)
    notes = s3_links_to_text(s3_urls)
    documents_df = docs_to_pandas(s3, docs_df["s3_url"], docs_df)
    documents_df.to_json('chartdata.json')
