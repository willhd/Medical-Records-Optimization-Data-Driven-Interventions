import re
import pandas as pd
import boto3
import pandas as pd
import numpy as np
import psycopg2
import string

"""extract specified features from corpus of text documents"""


class FeatureExtraction(object):
    def __init__(self, data):
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
        - lookup_race = retrieve race of patient from chart_note

        """
        self.chart_note = None
        self.data = data
        self.features = pd.DataFrame()

    def feature_dataframe(self, ID, DD, text):
        """returns dataframe with extracted features from corpus"""
        self.features['id'] = self.data[ID]
        self.features['doc_id'] = self.data[DD]
        self.features['dt'] = self.data[text].apply(self.lookup_visit_date)
        self.features['dx'] = self.data[text].apply(self.lookup_dx)
        self.features['age'] = self.data[text].apply(self.lookup_age)
        self.features['sex'] = self.data[text].apply(self.lookup_sex)
        self.features['race'] = self.data[text].apply(self.lookup_race)

    def clean_and_decode(self, read_column, write_column):
        """converts column of bytes to column of strings"""
        for i in range(len(self.data)):
            self.data[read_column][i] = self.data[write_column][i].decode("utf-8")

    def lookup_dx(self, chart_note):
        """returns list of diagnosis from a chart_note string"""
        d = re.findall("Diagnosis:.(.*?)\n\n", chart_note, flags=re.S | re.I)
        x = re.findall("medical history:.(.*?)\n\n", chart_note, flags=re.S | re.I)
        if len(d) >= 1:

            diags = [x.replace(',', '') for x in d]
            diags = [x.replace('\n', ',').strip() for x in diags]
            diags = [x.replace('\t', '') for x in diags]
            diags = [x.replace('    ', '') for x in diags]
            diags = diags[0].split(",")
            return [x.lower().strip('    ') for x in diags]
        elif len(x) >= 1:
            d = re.findall("history:.(.*?)\n\n", chart_note, flags=re.S | re.I)
            diags = [x.replace(',', '') for x in d]
            diags = [x.replace('\n', ',').strip() for x in diags]
            diags = [x.replace('\t', '') for x in diags]
            diags = [x.replace('    ', '') for x in diags]
            diags = diags[0].split(",")
            return [x.lower().strip('    ') for x in diags]
        else:
            return None

    def lookup_visit_date(self, chart_note):
        '''returns visit date as timestamp from chart_note'''
        dt = re.findall("date:.(.*?)\n", chart_note, flags=re.I)
        if len(dt) >= 1:
            return dt[0].strip()

    def lookup_age(self, chart_note):
        '''returns age as a string from a chart_note string'''
        age = re.findall("age:.(.*?)\n", chart_note, flags=re.I)
        if len(age) >= 1:
            return age[0].strip()

    def lookup_sex(self, chart_note):
        '''returns sex as a string from a chart_note string'''
        sex = re.findall("sex:.(.*?)\n", chart_note, flags=re.I)
        if len(sex) >= 1:
            return sex[0].strip()

    def lookup_race(self, chart_note):
        '''returns race as a string from a chart_note string'''
        race = re.findall("race:.(.*?)\n", chart_note, flags=re.I)
        if len(race) >= 1:
            return race[0].strip()


if __name__ == '__main__':
    Extraction = FeatureExtraction(data)
