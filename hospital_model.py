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
from sklearn.linear_model import LogisticRegression


def train_Multinomial(data="nb_data.json"):
    nb_df = pd.read_json(data)
    all_docs = nb_df["encounter-note"].values
    clf = MultinomialNB(alpha=0.01)
    b_tf = CountVectorizer(stop_words='english', tokenizer=xtokenizer)
    features = b_tf.fit(all_docs)
    f_transform = features.transform(all_docs)
    X = f_transform
    y = nb_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    Multinomial_model = Bclf.fit(X_train, y_train)
    return Multinomial_model


def score_Multinomial():
    Multinomial_model = train_Multinomial(data)
    y_hat = Multinomial_model.predict(X_test)
    y_hat_prob = Multinomial_model.predict_proba(X_test)
    print("acc score: {}".format(accuracy_score(y_test, y_hat)))
    print("prec score: {}".format(precision_score(y_test, y_hat)))
    print("recll score:{}".format(recall_score(y_test, y_hat)))


def plot_Multinomial(hard_or_soft, path):
    Multinomial_model = train_Multinomial(data)
    y_hat = Multinomial_model.predict(X_test)
    y_hat_prob = Multinomial_model.predict_proba(X_test)
    if hard_or_soft == "hard":
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(8, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plots the thresholds
        ax2 = plt.gca()
        ax2.plot(false_positive_rate, thresholds,
                 markeredgecolor='black', linestyle='dashed', color='red')
        ax2.set_ylabel('Threshold', color='red')
        ax2.set_ylim([thresholds[-1], thresholds[0]])
        ax2.set_xlim([false_positive_rate[0], false_positive_rate[-1]])
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(-0.1, 1.1)
        plt.savefig(path)
    else:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat_prob[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(8, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plots the thresholds
        ax2 = plt.gca()
        ax2.plot(false_positive_rate, thresholds,
                 markeredgecolor='black', linestyle='dashed', color='red')
        ax2.set_ylabel('Threshold', color='red')
        ax2.set_ylim([thresholds[-1], thresholds[0]])
        ax2.set_xlim([false_positive_rate[0], false_positive_rate[-1]])
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(-0.1, 1.1)
        plt.savefig(path)


def calculate_threshold_values(prob, y):
    '''
    returns profits associated with thresholds from various confusion-matrix
    ratios by threshold
    from a list of predicted probabilities and actual y values
    '''

    n_obs = float(len(y))

    predicted_probs = prob

    initial = [] if 1 in predicted_probs else [1]
    thresholds = initial + sorted(predicted_probs, reverse=True)
    cost_benefit = np.array([[2000, -150], [0, 0]])
    thresholds = np.arange(0, 1.1, 0.1)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs > threshold
        confusion_matrix = standard_confusion_matrix(y, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit)
        profits.append(threshold_profit / n_obs)
    return thresholds, profits


def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    from two 1D arrays 
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def profit_curve():
    fig, ax = plt.subplots(figsize=(10, 10))

    xlog, ylog = calculate_threshold_values(y_hat_prob[:, 1], y_test)
    x, y = calculate_threshold_values(yhat_log_prob[:, 1], y_test)

    ax.plot(xlog, ylog)
    ax.plot(x, y)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('profits')
    ax.set_title('Profit Curve')
    ax.set_xlim(xmin=0, xmax=1)


def train_Bernoulli(data="nb_data.json"):
    nb_df = pd.read_json(data)
    all_docs = nb_df["encounter-note"].values
    Bclf = BernoulliNB(alpha=0.01)
    b_tf = CountVectorizer(stop_words='english', tokenizer=xtokenizer)
    features = b_tf.fit(all_docs)
    f_transform = features.transform(all_docs)
    X = f_transform
    y = nb_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    Bernoulli_model = Bclf.fit(X_train, y_train)
    return Bernoulli_model


def score_Bernoulli():
    Bernoulli_model = train_Bernoulli(data)
    yhat_b = Bernoulli_model.predict(X_test)
    yhat_b_prob = Bernoulli_model.predict_proba(X_test)
    print("accuracy_score : {}".format(accuracy_score(y_test, yhat_b)))
    print("precision_score: {}".format(precision_score(y_test, yhat_b)))
    print("recall_score:{}".format(recall_score(y_test, yhat_b)))


def plot_Bernoulli(hard_or_soft, path):
    Bernoulli_model = train_Bernoulli(data)
    yhat_b = Bernoulli_model.predict(X_test)
    yhat_b_prob = Bernoulli_model.predict_proba(X_test)
    if hard_or_soft == "hard":
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, yhat_b)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(8, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plots the thresholds
        ax2 = plt.gca()
        ax2.plot(false_positive_rate, thresholds,
                 markeredgecolor='black', linestyle='dashed', color='red')
        ax2.set_ylabel('Threshold', color='red')
        ax2.set_ylim([thresholds[-1], thresholds[0]])
        ax2.set_xlim([false_positive_rate[0], false_positive_rate[-1]])
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(-0.1, 1.1)
        plt.savefig(path)
    else:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, yhat_b_prob[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(8, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plots the thresholds
        ax2 = plt.gca()
        ax2.plot(false_positive_rate, thresholds,
                 markeredgecolor='black', linestyle='dashed', color='red')
        ax2.set_ylabel('Threshold', color='red')
        ax2.set_ylim([thresholds[-1], thresholds[0]])
        ax2.set_xlim([false_positive_rate[0], false_positive_rate[-1]])
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlim(-0.1, 1.1)
        plt.savefig(path)
