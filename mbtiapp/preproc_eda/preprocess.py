# -*- coding: utf-8 -*-

# nohup python /Users/a/mbti/mbtiapp/preproc_eda/preprocess.py --src_dat_path='/Users/a/mbti/mbtidat/collection/dat/mbti_raw.parquet' > /Users/a/mbti/mbtiapp/preproc_eda/preprocess.out 2>&1 &

import os
import sys
import pandas as pd
from pathlib import Path
import argparse

from text_cleaner import TextCleaner

parser = argparse.ArgumentParser()
parser.add_argument('--src_dat_path', type=str, default=None)

common_func_dir = os.path.abspath((Path(__file__).parent / '../../common_func').resolve())
sys.path.append(common_func_dir)

from config import PREPROC_EDA_DAT_DIR, PREPROC_EDA_OUT_DIR
from logger import JsonLogger


def load_dataset(src_path):
    extension = os.path.splitext(src_path)[-1]
    if extension == '.parquet':
       	df = pd.read_parquet(src_path, engine='pyarrow')
    elif extension == '.csv':
        df = pd.read_csv(src_path)
    else:
        raise Exception('Invalid extension({}).'.format(extension))
    return df

def export_dataset(df, dst_dir, file_id, dtype='parquet_csv'):

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    dst_path_parquet = os.path.join(dst_dir, file_id + '.parquet')
    dst_path_csv = os.path.join(dst_dir, file_id + '.csv')
    if dtype == 'parquet':
        df.to_parquet(dst_path_parquet, engine='pyarrow', index=False)
    elif dtype == 'csv':
        df.to_csv(dst_path_csv, index=False)
    else:
        df.to_parquet(dst_path_parquet, engine='pyarrow', index=False)
        df.to_csv(dst_path_csv, index=False)


# Warning: stop_word를 일부라도 포함하는 단어는 통째로 제거됨, Tokenize 전 단계라 중요한 단어가 지워질 수 있음
def get_mbti_stop_words(stop_words=[]):

    enneagram_words = ['1w9', '1w2', '2w1', '2w3', '3w2', '3w4', '4w3', '4w5', '5w4', '5w6', '6w5', '5w6', '6w5', '6w7', '7w6', '7w8', '8w7', '8w9', '9w8', '9w1']
    mbti_words = ['enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp', 'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp']

    stop_words.extend(enneagram_words)
    stop_words.extend(mbti_words)

    return list(set(stop_words))

def remove_irrelevant_boards(df, board_column='board'):

    irrelevant_boards = ['가입 인사 (등업 필수)', '바라의 자유문답 게시판', '가람의 100문 100답', '새옴의 20문 20답', '심리체질의학 (창천)', '서평 이벤트 (구)', '[종합] 모임 단톡 공지', '심리 자료실']

    df = df[df[board_column].isin(irrelevant_boards) == False]

    return df

def label_y_val(y_val_raw):

    y_val_list = ['enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp', 'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp']
    target = total_target = ''
    result_dict = {}

    try:
        for word in y_val_list:
            idx = y_val_raw.lower().find(word)
            if idx >= 0:
                result_dict[word] = idx

        res = sorted(result_dict.items(), key=(lambda x: x[1]))

        total_list = []
        
        for cnt, word in enumerate(res):
            if cnt == 0:
                target = word[0]
            total_list.append(word[0])
                
        total_target = ','.join(total_list)
    except:
        pass

    return target, total_target

def normalize_date(date_raw):
    date_raw_list = date_raw.split('.')

    try:
        date = str(date_raw_list[0]).rjust(4, '0') + '-' + str(date_raw_list[1]).rjust(2, '0') + '-' + str(date_raw_list[2]).rjust(2, '0')
    except:
        date = ''
    return date


def main():

    # raw data 로드
    df = load_dataset(opt.src_dat_path)

    # 중복 게시글 제거
    df = df.drop_duplicates(['article_num'], keep='first')

    # 가입 인사, 설문 조사, 공지 사항 등 게시판 글 제거
    df = remove_irrelevant_boards(df, board_column='board')

    # mbti 및 기타 성격 분류 등 직접적으로 mbti를 나타내는 어휘 제거
    stop_words = get_mbti_stop_words()
    cleaner = TextCleaner(stop_words)

    # 게시글 제목에 대해 특수문자, 이모티콘 등 제거
    df['title_cleaned'] = df['title'].apply(lambda x: cleaner.cleanup_text(x))

    # 제시글 내용에 대해 특수문자, 이모티콘 등 제거
    df['contents_cleaned'] = df['contents'].apply(lambda x: cleaner.cleanup_text(x))

    # 게시글 텍스트 길이가 30 이상 4000 미만인 것들만 필터, 너무 짧은 글은 유효하지 않고, 너무 긴글은 자신의 글이 아니라 다른곳에서 퍼온글이거나 분석글일 가능성이 높음
    df['contents_cleaned'] = df['contents_cleaned'].apply(lambda x: x if len(x) > 30 and len(x) < 4000 else '')
    #df['contents_cleaned'] = df['contents_cleaned'].apply(lambda x: x if len(x) > 30 else '')
    df = df[df['contents_cleaned'] != '']

    df['X_val'] = df['title_cleaned'] + ' ' + df['contents_cleaned']

    # 날짜 'YYYY-mm-dd' format으로 normalization
    df['date_cleaned'] = df['date'].apply(lambda x: normalize_date(x))

    # nick_name으로 부터 target 추출
    df['target_desc'], df['total_target'] = zip(*df['nick_name'].apply(lambda x: label_y_val(x)))
    df = df[df['target_desc'] != '']

    mapping = {'enfj': 0, 'enfp': 1, 'entj': 2, 'entp': 3,
                'esfj': 4, 'esfp': 5, 'estj': 6, 'estp': 7,
                'infj': 8, 'infp': 9, 'intj': 10, 'intp': 11,
                'isfj': 12, 'isfp': 13, 'istj': 14, 'istp': 15}

    df['target'] = df['target_desc'].map(mapping)

    map1 = {"i": 0, "e": 1}
    map2 = {"n": 0, "s": 1}
    map3 = {"t": 0, "f": 1}
    map4 = {"j": 0, "p": 1}
    df['i_e'] = df['target_desc'].astype(str).str[0]
    df['i_e'] = df['i_e'].map(map1)
    df['n_s'] = df['target_desc'].astype(str).str[1]
    df['n_s'] = df['n_s'].map(map2)
    df['t_f'] = df['target_desc'].astype(str).str[2]
    df['t_f'] = df['t_f'].map(map3)
    df['j_p'] = df['target_desc'].astype(str).str[3]
    df['j_p'] = df['j_p'].map(map4)

    file_id = 'mbti_cleaned_df'

    # parquet, csv format으로 전처리된 데이터 저장
    export_dataset(df, PREPROC_EDA_DAT_DIR, file_id)

if __name__ == '__main__':

    opt = parser.parse_args()

    main()
