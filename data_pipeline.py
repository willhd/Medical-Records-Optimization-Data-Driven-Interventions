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
import string
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser
from nltk.tag.stanford import NERTagger


def get_chart_data(s3_client, Bucket, Key):
    """open s_3 bucket get the data and return the text"""
    s3_client = start_session()
    with s3_client.get_object(Bucket=Bucket, Key=Key) as chart_object:
        chart_data = chart_object['Body'].read()
    return chart_data


def silhouette(sparse_matrix, cluster_range):
    """Returns silhouette coefficient for a provided number of clusters
    Parameters
    ----------
    sparse_matrix : tfidf numpy array
    -------
    cluster_range (int): the number of clusters to to calculate
    silhoutee score for

    """

    X = sparse_matrix
    for n_cluster in range(cluster_range):
    kmeans = KMeans(n_clusters=n_cluster + 1).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


def get_top_five(X, transform_content, n_clusters=20, max_features=None):
    content_means = KMeans(n_clusters=n_clusters)
    content_means.fit(transform_content)
    top_five = np.argsort(content_means.cluster_centers_, axis=1)[:, -20:]
    for i in range(top_five.shape[0]):
        print(np.array(tfidf.get_feature_names())[top_five[i]])


def extract_bow_from_raw_text(text_as_string):
    """Extracts bag-of-words from a raw text string, and removes names.
    Parameters
    ----------
    text (str): a text document given as a string
    Returns
    -------
    list : the list of the tokens extracted and filtered from the text
    """
    if (text_as_string == None):
        return []

    if (len(text_as_string) < 1):
        return []

    nfkd_form = unicodedata.normalize('NFKD', text_as_string)
    text_input = str(nfkd_form.encode('ASCII', 'ignore'))

    sent_tokens = sent_tokenize(text_input)
        st = NERTagger('stanford-ner/all.3class.distsim.crf.ser.gz',
                       'stanford-ner/stanford-ner.jar')
        for sent in sent_tokens:
        tags = st.tag(tokens)
        for tag in tags:
            if tag[1] == 'PERSON':
                tokens.del(tag)

    tokens = list(map(word_tokenize, sent_tokens))

    sent_tags = list(map(pos_tag, tokens))

    grammar = r"""
        SENT: {<(J|N).*>}
    """

    cp = RegexpParser(grammar)
    ret_tokens = list()
    stemmer_snowball = SnowballStemmer('english')
    import nltk

    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'SENT':
                t_tokenlist = [tpos[0].lower() for tpos in subtree.leaves()]
                t_tokens_stemsnowball = list(map(stemmer_snowball.stem, t_tokenlist))
                ret_tokens.extend(t_tokens_stemsnowball)
    return(ret_tokens)


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
