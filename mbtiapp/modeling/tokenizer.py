# -*- coding: utf-8 -*-

import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from konlpy.tag import Okt
from konlpy.tag import Kkma
from mecab import MeCab

class Tokenizer:
	
    def __init__(self, stopwords):

        self._stopwords = stopwords

    def tokenize_by_nltk(self, doc):

        return [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(doc)) if not word in self._stopwords and pos[0] == 'N' and len(word) != 1]

    def tokenize_by_mecab(self, doc):

        m = MeCab()
        malist = m.nouns(doc)

        return [word for word in malist if not word in self._stopwords and len(word) != 1]

    def tokenize_by_okt(self, doc):

        t = Okt()
        malist = t.nouns(doc)

        return [word for word in malist if not word in self._stopwords and len(word) != 1]

    def tokenize_by_kkma(self, doc):
	
        k = Kkma()
        malist = k.nouns(doc)

        return [word for word in malist if not word in self._stopwords and len(word) != 1]
