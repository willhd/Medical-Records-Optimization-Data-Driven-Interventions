import re
import pandas as pd
import boto3
import pandas as pd
import numpy as np
import psycopg2
import string

"""extract specified features from corpus of text documents"""


class FeatureExtraction(object):
    def __init__(self, data, suffix, profile_name, Bucket, s3_client):
        """
        INPUT:
        - data = Path to data file as JSON string

        ATTRIBUTES:
        - Data = pandas dataframe converted from json string with document column
        - chart_note = a medical document as a string
        - lookup_dx = retrieve problem list with ICD codes from chart_note
        - lookup_visit_date = retrieve visit date of patient from chart_note
        - lookup_age = retrieve age of patient from chart_note
        - lookup_sex = retrieve gender of patient from chart_note
        - suffix

        """
        self.chart_note = None
        self.data = pd.read_json(data)
        self.features = pd.DataFrame(columns=['id', 'doc_id', 'dx', 'age', 'sex', 'dt'])
        self.suffix = suffix
        self.s3 = None
        self.Bucket = Bucket
        self.profile_name = profile_name
        self.s3_client = s3_client

    def feature_dataframe(self, ID, text):
        """returns dataframe with extracted features from corpus"""
        self.features['id'] = self.data[ID]
        #self.features['doc_id'] = self.data['doc_id']
        self.features['dx'] = self.data[text].apply(self.lookup_dx)
        self.features['age'] = self.data[text].apply(self.lookup_age)
        self.features['sex'] = self.data[text].apply(self.lookup_sex)
        self.features['dt'] = self.data[text].apply(self.lookup_visit_date)

    def start_session(self):
        ''' start an s3 session with boto3, returns s3 session object'''
        session = boto3.Session(profile_name=self.profile_name)
        s3 = session.client(self.s3_client)
        self.s3 = s3

    def get_chart_data(self, Key):
        """ open s3_bucket, get the data, close the streaming text body,
        and return the text data """
        chart_object = self.s3.get_object(Bucket=self.Bucket, Key=Key)
        chart_body = chart_object['Body']
        chart_data = chart_body.read()
        chart_body.close()
        return chart_data

    def docs_to_pandas(self, read_column, write_column):
        """returns pandas dataframe with text column from s3_url"""
        self.data.dropna(axis=0, how='any', inplace=True)
        self.data.reset_index(inplace=True)
        s3_urls = self.data[read_column].values
        texts = []
        for s3_url in s3_urls:
            offset = len(self.suffix)
            Key = s3_url[offset:]
            Bucket = self.Bucket
            raw_text = self.get_chart_data(Key=Key)
            texts.append(raw_text)
        self.data[write_column] = texts

    def clean_and_decode(self, read_column, write_column):
        """converts column of bytes to column of strings"""
        for i in range(len(self.data)):
            self.data[read_column][i] = self.data[write_column][i].decode("utf-8")

    def lookup_dx(self, chart_note):
        """returns list of diagnosis from a chart_note string"""
        d = re.findall("Diagnosis:.(.*?)\n\n", chart_note, flags=re.S | re.I)
        if d != None:

            diags = [x.replace(',', '') for x in d]
            diags = [x.replace('\n', ',').strip() for x in diags]
            diags = [x.replace('\t', '') for x in diags]
            diags = [x.replace('    ', '') for x in diags]
            diags = diags[0].split(",")
            return [x.lower().strip('    ') for x in diags]
        elif d == None:
            d = re.findall("history:.(.*?)\n\n", chart_note, flags=re.S | re.I)
            diags = [x.replace(',', '') for x in d]
            diags = [x.replace('\n', ',').strip() for x in diags]
            diags = [x.replace('\t', '') for x in diags]
            diags = [x.replace('    ', '') for x in diags]
            diags = diags[0].split(",")
            return [x.lower().strip('    ') for x in diags]

    def lookup_visit_date(self, chart_note):
        '''returns visit date as timestamp from chart_note'''
        dt = re.findall("date:.(.*?)\n", chart_note, flags=re.I)
        return dt[0].strip()

    def lookup_age(self, chart_note):
        '''returns age as a string from a chart_note string'''
        age = re.findall("age:.(.*?)\n", chart_note, flags=re.I)
        return age[0].strip()

    def lookup_sex(self, chart_note):
        sex = re.findall("sex:.(.*?)\n", chart_note, flags=re.I)
        return sex[0].strip()


if __name__ == '__main__':
    """
    initialize with s3_credentials 
    extraction = FeatureExtraction("path_to_json_string",
                                   's3_suffix',
                                   's3_profile_name', 's3_bucket_name',
                                  's3_client')
                                  """
