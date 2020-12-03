# -*- coding: utf-8 -*-

import re

class TextCleaner:
	
    def __init__(self, stop_words=[]):
        
        self._stop_words = stop_words

    def cleanup_text(self, text):
			
        rules = [
            {r'[\n\t\r]': u' '}, # remove control charactor
            {r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)': u' '}, # remove email
            {r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+': u' '}, # remove url
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': u' '}, # remove link
            {r'<[^>]*>': u' '}, # remove tag
            {r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]': u' '}, # remove special character
            {r'([ㅋ][ㅋ]+|[ㅎ][ㅎ]+|[k][k]+|[z][z]+)': u' 웃음 '}, # ㅋㅋㅋㅋ, ㅎㅎ, zzz, kkk -> 웃음 변환
            {r'([ㅜ]+|[ㅠ]+)': u' 슬픔 '}, # ㅜ, ㅠㅠ -> 슬픔 변환
            {r'([ㄱ-ㅎㅏ-ㅣ]+)': u' '}, # remove kor consonant vowel
            {r'[0-9]+': u' '} # remove number
        ]

        self._stop_words.extend(['은', '는', '이', '가'])
        stop_words_rule = {r'|'.join(self._stop_words): u' '}
        rules.append(stop_words_rule)
        
        for rule in rules:
            for key, val in rule.items():
                text = text.strip().lower()
                reg = re.compile(key)
                text = reg.sub(val, text)
                text = " ".join([t for t in text.split() if len(t) > 1])
        return text.strip()

    def remove_all_except_kor(self, text):
			
        rules = [
            {r'[^ ㄱ-ㅣ가-힣]+': u' '}, # remove all except kor
            {r'([ㅋ][ㅋ]+|[ㅎ][ㅎ]+)': u' 웃음 '}, # ㅋㅋㅋㅋ, ㅎㅎ -> 웃음 변환
            {r'([ㅜ]+|[ㅠ]+)': u' 슬픔 '} # ㅜ, ㅠㅠ -> 슬픔 변환
        ]

        self._stop_words.extend(['은', '는', '이', '가'])      
        stop_words_rule = {r'|'.join(self._stop_words): u' '}
        rules.append(stop_words_rule)
        
        for rule in rules:
            for key, val in rule.items():
                text = text.strip().lower()
                reg = re.compile(key)
                text = reg.sub(val, text)
                text = " ".join([t for t in text.split() if len(t) > 1])
        return text.strip()
