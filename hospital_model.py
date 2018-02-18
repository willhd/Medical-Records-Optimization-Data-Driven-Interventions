import string
import re
import pandas as pd
import numpy as np
import boto3
import json
import pickle
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, silhouette_samples, silhouette_score, confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression


class HospitalModel(object)
    def __init__(self, data, alpha=0.1, n_jobs=-1, max_features='sqrt',
                 NaiveBayes=True, LogisticRegression=True):
        """
        INPUT:
        - n_jobs = Number of jobs to run models on
        - max_features = Number of features to consider for CountVectorizer, default is 'sqrt'
        - NaiveBayes = Bool, run MNB
        - LogisticRegression = Bool, run LogR
        - data = Path to data file as JSON string

        ATTRIBUTES:
        - LogR= Logistic Regression Classifier
        - MNB = Multinomial Naive Bayes Classifier
        - Data = pandas dataframe converted from json string with document column
        """
        self.xtokenizer = xtokenizer
        self.names = None  # set of names to remove during tokenization step
        self.CV = CountVectorizer(n_jobs=n_jobs, max_features=max_features,
                                  stop_words='english', tokenizer=self.xtokenizer)
        self.MNB = MultinomialNB(alpha=alpha)
        self.LogR = LogisticRegression(penalty='l2', C=0.5, max_iter=100000, class_weight={0: 0.07, 1: 0.93}, random_state=22, solver='sag'))
        self.Data=None
        self.TF=None
        self.Target=None
        self.Train=None
        self.X_train
        self.X_test
        self.y_train
        self.y_test


    def train_model(self, column):
        """fits model to specified column of raw text documents"""
        corpus=pd.read_json(self.Data)
        features=self.CV.fit(corpus)
        self.TF=features.transform(corpus)
        X=self.TF
        y=self.Data[column]
        self.Target=y
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(
            X, y, test_size = 0.33, random_state = 42)
        if self.NaiveBayes == True:
            self.MNB.fit(X_train, y_train)
        if self.LogisticRegression == True:
            self.LogR.fit(X_train, y_train)


    def cross_val_score(self, model, cv):
        "scores model with cross validation with cv folds"
        scores=cross_val_score(self.model, self.TF, cv = cv)
        return scores.mean()

    def model_score(self):
        """scores classification model on test set.

        report includes: precision, recall, f1-score, and support for both groups 0 and 1.

        """
        yhat=self.Train.predict(self.X_test)
        yhat_prob=self.Train.predict_proba(self.X_test)
        return classification_report(self.y_test, yhat)


    def plot_model(self, path, thresholds = False):
        """plots ROC_AUC at designated path"""
        false_positive_rate, true_positive_rate, thresholds=roc_curve(y_test, y_hat_prob[:, 1])
        roc_auc=auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize = (8, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], color = 'black', linestyle = 'dashed')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plots the thresholds on twin axis
        if thresholds == True:
            ax2=plt.gca()
            ax2.plot(false_positive_rate, thresholds,
                     markeredgecolor = 'black', linestyle = 'dashed', color = 'red')
            ax2.set_ylabel('Threshold', color = 'red')
            ax2.set_ylim([thresholds[-1], thresholds[0]])
            ax2.set_xlim([false_positive_rate[0], false_positive_rate[-1]])
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlim(-0.1, 1.1)
        plt.savefig(path)


    def calculate_threshold_values(self, prob, y, cost_benefit):
        '''
        returns profit values associated with thresholds from various confusion-matrix
        ratios by threshold

        from a list of predicted probabilities and actual y values,

        the cost_benefit matrix argument is a numpy array for example: cost_benefit=np.array([[2000, -150], [0, 0]])
        '''

        n_obs=float(len(y))

        predicted_probs=prob

        initial=[] if 1 in predicted_probs else [1]
        thresholds=initial + sorted(predicted_probs, reverse = True)

        cost_benefit=cost_benefit
        thresholds=np.arange(0, 1.1, 0.1)
        profits=[]
        for threshold in thresholds:
            y_predict=predicted_probs > threshold
            confusion_matrix=standard_confusion_matrix(y, y_predict)
            threshold_profit=np.sum(confusion_matrix * cost_benefit)
            profits.append(threshold_profit / n_obs)
        return thresholds, profits


    def standard_confusion_matrix(self, y_true, y_pred):
        """Connverts sci-kit leaen confusion matrix to standard matrix with format:
                      -----------
                      | TP | FP |
                      -----------
                      | FN | TN |
                      -----------
        from two 1D arrays
        """
        [[tn, fp], [fn, tp]]=confusion_matrix(y_true, y_pred)
        return np.array([[tp, fp], [fn, tn]])


    def profit_curve(self):
        '''plots the profit curve for the associated cost-bennifit and confusion_matrix'''
        fig, ax=plt.subplots(figsize = (10, 10))

        xlog, ylog=calculate_threshold_values(y_hat_prob[:, 1], y_test)
        x, y=calculate_threshold_values(yhat_log_prob[:, 1], y_test)

        ax.plot(xlog, ylog)
        ax.plot(x, y)
        ax.set_xlabel('thresholds')
        ax.set_ylabel('profits')
        ax.set_title('Profit Curve')
        ax.set_xlim(xmin = 0, xmax = 1)

    def nametokens(self, names):
        self.names=set(names)

    def xtokenizer(self, text):
        return [xnumbers(word) for word in self.tokenizer(text)]

    def xnumbers(self, word):
        if word.startswith("0") and ":" not in word and "/" not in word:
            output=''
            for letter in word:
                if letter == "0":
                    output += "0"
                else:
                    break
            while len(output) < len(word):
                output += 'x'
            return output
        else:
            # return word
            if word in self.names:
                return ''
            else:
                return word

    def tokenizer(text, token_pattern = r"(?u)\b\w\w+\b"):
        token_pattern=re.compile(token_pattern)
        return token_pattern.findall(text)
