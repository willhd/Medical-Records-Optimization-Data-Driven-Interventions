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

"""This class fits LDA model to raw text documents and provides methods for exploring latent features"""


class EncounterTypes(data, LDA=True):
    def __init__(self, n_jobs=-1, max_features='sqrt',
                 NMF=True, LDA=True):
        """
        INPUT:
        - n_jobs = Number of jobs to run models on
        - max_features = Number of features to consider for CountVectorizer, default is 'sqrt'
        - NMF = Bool, run NMF
        - LDA = Bool, run LDA

        ATTRIBUTES:
        - LdaModel= Logistic Regression Classifier
        - NmfModel = Multinomial Naive Bayes Classifier
        - Data = json string with document column
        """
        self.LdaModel = None
        self.NmfModel = None
        self.Data = Encounter
        self.TOPICS = None

    def label_counts(self, n):
        notes = []
        for i in range(len(all_documents)):
            if np.argmax(predictions_lda[i][:]) == n:
                notes.append(i)
        return len(notes)

    def find_notes_labeled(self, n):
        notes = []
        for i in range(len(all_documents)):
            if np.argmax(predictions_lda[i][:]) == n:
            notes.append(i)
        return notes

    def encounter_label(self, label):
        doc = all_documents[label].strip()
        lbl = np.argmax(predictions_lda[label][:])
        # return (lbl,doc.strip())
        print(lbl)
        print(doc.strip())

    def display_topics(self, model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    LDA_TOPICS = dict({0: "Lab Results",
                       1: "Screening/ Refferal",
                       2: "Speciality Visit",
                       3: "Social History",
                       4: "Treatment Plan",
                       5: "Insurance/ Billing",
                       6: "Mental Health",
                       7: "Lab Results",
                       8: "Summary View",
                       9: "Problem focused visit",
                       10: "Prevention plan",
                       11: "Comprehensive View",
                       12: "Patient Communication"})

    NMF_TOPICS = dict({0: "Comprehensive Diagnostic",
                       1: "Patient Communication ",
                       2: "Summary view",
                       3: "Comprehensive view",
                       4: "Screening",
                       5: "Problem focused visit",
                       6: "Medications/Tox-screen",
                       7: "Primary Care",
                       8: "Insurance/ Billing",
                       9: "Inpatinet/Specialty visit",
                       10: "Comprehensive Lab Resuls",
                       11: "Specific Lab Order",
                       12: "Referral/ Care Coordination"})
