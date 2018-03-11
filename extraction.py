import re
import pandas as pd

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

        """
        self.chart_note = None
        self.data = pd.read_json(data)
        self.features = pd.DataFrame(columns=['id', 'dx', 'age', 'sex', 'dt'])

    def lookup_dx(self, chart_note):
        """returns list of diagnosis from a chart_note string"""
        d = re.findall("Diagnosis:.(.*?)\n\n", chart_note, flags=re.S | re.I)
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
        sex = re.findall("sex:.(.*?)\DOB", chart_note, flags=re.I)
        return sex[0].strip()
