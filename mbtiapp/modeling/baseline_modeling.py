# -*- coding: utf-8 -*-

# nohup python /Users/a/mbti/mbtiapp/modeling/baseline_modeling.py --src_dat_path /Users/a/mbti/mbtidat/preproc_eda/dat/mbti_cleaned_df.parquet > /Users/a/mbti/mbtiapp/modeling/baseline_modeling.out 2>&1 &

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tendo import singleton
from datetime import datetime
from imblearn.combine import SMOTEENN
from sklearn.feature_extraction.text import CountVectorizer

from simple_classifier import SimpleNLClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--src_dat_path', type=str, default=None)

common_func_dir = os.path.abspath((Path(__file__).parent / '../../common_func').resolve())
sys.path.append(common_func_dir)

from config import MODELING_DAT_DIR, MODELING_OUT_DIR

def load_dataset(src_path):
    extension = os.path.splitext(src_path)[-1]
    if extension == '.parquet':
        df = pd.read_parquet(src_path, engine='pyarrow')
    elif extension == '.csv':
        df = pd.read_csv(src_path)
    else:
        raise Exception('Invalid extension({}).'.format(extension))
    return df

def BOW(data, X_col='contents_cleaned', y_col='target'):
    
    df_temp = data.copy(deep = True)

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(df_temp[X_col])

    list_corpus = df_temp[X_col].tolist()
    list_labels = df_temp[y_col].tolist()
    
    X = count_vectorizer.transform(list_corpus)
    
    return X, list_labels

def main():

        stop_words = []
        enneagram_words = ['1w9', '1w2', '2w1', '2w3', '3w2', '3w4', '4w3', '4w5', '5w4', '5w6', '6w5', '5w6', '6w5', '6w7', '7w6', '7w8', '8w7', '8w9', '9w8', '9w1']
        mbti_words = ['enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp', 'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp']

        stop_words.extend(enneagram_words)
        stop_words.extend(mbti_words)
	
        df = load_dataset(opt.src_dat_path)

        X_data = df['contents_cleaned']
        y_data = df['target']
        X_data, y_data = BOW(df)

        X_data_bal, y_data_bal = SMOTEENN(random_state=0).fit_sample(X_data, y_data)
        sc = SimpleNLClassifier(X_data_bal, y_data_bal, stop_words)

        X_train, X_test, y_train, y_test = sc.train_test_split(test_size=0.3, seed=7)

        dt_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        dst_path = os.path.join(MODELING_OUT_DIR, 'classification_report_{}.txt'.format(dt_str))

        #multinomial_nb = sc.multinomial_naive_bayes(X_train, y_train, max_features=50000)
        #multinomial_nb_predicted = sc.predict(multinomial_nb, X_test)
        #sc.print_report(y_test, multinomial_nb_predicted, dst_path, method='Multinomial Naive Bayes')

        #k_neighbors = sc.k_neighbors(X_train, y_train, max_features=50000)
        #k_neighbors_predicted = sc.predict(k_neighbors, X_test)
        #sc.print_report(y_test, k_neighbors_predicted, dst_path, method='K Neighbors')
	
        #linear_svm = sc.linear_svm(X_train, y_train, max_features=50000)
        #linear_svm_predicted = sc.predict(linear_svm, X_test)
        #sc.print_report(y_test, linear_svm_predicted, dst_path, method='Linear SVM')

        #random_forest = sc.random_forest(X_train, y_train, max_features=50000, n_estimators=500)
        #random_forest_predicted = sc.predict(random_forest, X_test)
        #sc.print_report(y_test, random_forest_predicted, dst_path, method='Random Forest')

        xgboost = sc.xgboost(X_train, y_train, max_features=80000, max_depth=10, n_estimators=500, learning_rate=0.1)
        xgboost_predicted = sc.predict(xgboost, X_test)
        sc.print_report(y_test, xgboost_predicted, dst_path, method='XGBoost')

        X_data = df['contents_cleaned']
        y_data = df['i_e']
        X_data, y_data = BOW(df, X_col='contents_cleaned', y_col='i_e')
        X_data_bal, y_data_bal = SMOTEENN(random_state=0).fit_sample(X_data, y_data)
        sc = SimpleNLClassifier(X_data_bal, y_data_bal, stop_words)

        X_train, X_test, y_train, y_test = sc.train_test_split(test_size=0.3, seed=7)

        dt_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        dst_path = os.path.join(MODELING_OUT_DIR, 'classification_report_{}.txt'.format(dt_str))

        xgboost = sc.xgboost(X_train, y_train, max_features=80000, max_depth=10, n_estimators=500, learning_rate=0.1)
        xgboost_predicted = sc.predict(xgboost, X_test)
        sc.print_report(y_test, xgboost_predicted, dst_path, method='XGBoost')


        X_data = df['contents_cleaned']
        y_data = df['n_s']

        sc = SimpleNLClassifier(X_data, y_data, stop_words)

        X_train, X_test, y_train, y_test = sc.train_test_split(test_size=0.3, seed=7)

        dt_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        dst_path = os.path.join(MODELING_OUT_DIR, 'classification_report_{}.txt'.format(dt_str))

        #xgboost = sc.xgboost(X_train, y_train, max_features=80000, max_depth=10, n_estimators=500, learning_rate=0.1)
        #xgboost_predicted = sc.predict(xgboost, X_test)
        #sc.print_report(y_test, xgboost_predicted, dst_path, method='XGBoost')

        X_data = df['contents_cleaned']
        y_data = df['t_f']

        sc = SimpleNLClassifier(X_data, y_data, stop_words)

        X_train, X_test, y_train, y_test = sc.train_test_split(test_size=0.3, seed=7)

        dt_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        dst_path = os.path.join(MODELING_OUT_DIR, 'classification_report_{}.txt'.format(dt_str))

        #xgboost = sc.xgboost(X_train, y_train, max_features=80000, max_depth=10, n_estimators=500, learning_rate=0.1)
        #xgboost_predicted = sc.predict(xgboost, X_test)
        #sc.print_report(y_test, xgboost_predicted, dst_path, method='XGBoost')

        X_data = df['contents_cleaned']
        y_data = df['j_p']

        sc = SimpleNLClassifier(X_data, y_data, stop_words)

        X_train, X_test, y_train, y_test = sc.train_test_split(test_size=0.3, seed=7)

        dt_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        dst_path = os.path.join(MODELING_OUT_DIR, 'classification_report_{}.txt'.format(dt_str))

        #xgboost = sc.xgboost(X_train, y_train, max_features=80000, max_depth=10, n_estimators=500, learning_rate=0.1)
        #xgboost_predicted = sc.predict(xgboost, X_test)
        #sc.print_report(y_test, xgboost_predicted, dst_path, method='XGBoost')

if __name__ == '__main__':

    try:
        me = singleton.SingleInstance()
    except:
        sys.exit(-1)

    opt = parser.parse_args()

    main()