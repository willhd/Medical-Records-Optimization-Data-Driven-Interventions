import string
import re
import pandas as pd
import numpy as np
import boto3
import json
import pickle
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
%matplotlib inline
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
from nltk import nr_chunk


def count_labels(label):
    labels = []
    for i in kmeans_thirty_model.labels_:
        if i == label:
            labels.append(i)
    return len(labels)


def extract_entities(text):
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'node'):
                print chunk.node, ' '.join(c[0] for c in chunk.leaves())


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def xnumbers(word):
    if word.startswith("0") and ":" not in word and "/" not in word:
        output = ''
        for letter in word:
            if letter == "0":
                output += "0"
            else:
                break
        while len(output) < len(word):
            output += 'x'
        return output
    else:
        return word


def xtokenizer(text):
    return [xnumbers(word) for word in tokenizer(text)]


def get_top_thirty(content_means):
    #content_means = KMeans(n_clusters=n_clusters)
    # content_means.fit(transform_content)
    top_thirty = np.argsort(content_means.cluster_centers_, axis=1)[:, -30:]
    for i in range(top_thirty.shape[0]):
        print(np.array(tfidf.get_feature_names())[top_thirty[i]])


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


def get_top_five(transform_content, n_clusters, max_features=None):
    """Extracts top five words from each cluster
    ----------
    n_clusters (int): specifies number of k-means clusters
    -------
    transform_content : sparse_matrix of word counts
    """
    content_means = KMeans(n_clusters=n_clusters)
    content_means.fit(transform_content)
    top_five = np.argsort(content_means.cluster_centers_, axis=1)[:, -5:]
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
