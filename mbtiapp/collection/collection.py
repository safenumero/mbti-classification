# -*- coding: utf-8 -*-

import os
import sys
import time
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
import pyperclip
from pathlib import Path

common_func_dir = os.path.abspath((Path(__file__).parent / "../../common_func").resolve())
sys.path.append(common_func_dir)

from config import NAVER_CONFIG, COLLECTION_DAT_DIR


def get_driver():
    
    retried = 0; max_attempts = 5
    while (retried < max_attempts):
        try:
            ua = UserAgent(verify_ssl=False)
            options = Options()
            options.add_argument('--user-agent={}'.format(ua.random))
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument("--lang=ko_KR")
            driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
            driver.implicitly_wait(30)
            break
        except:
            retried += 1
            time.sleep(1)

    if retried >= max_attempts:
        raise Exception('Failed to get chrome driver.')
        
    driver.implicitly_wait(3)

    return driver

def naver_login(driver, naver_id, naver_passwd):
    
    url = 'https://nid.naver.com/nidlogin.login?svctype=262144&url=http://m.naver.com/'
    driver.get(url)
    driver.implicitly_wait(3)
    driver.execute_script("document.getElementsByName('id')[0].value=\'" + naver_id + "\'")
    time.sleep(1) 
    driver.execute_script("document.getElementsByName('pw')[0].value=\'" + naver_passwd + "\'")
    time.sleep(1) 
    driver.find_element_by_xpath('//*[@id="frmNIDLogin"]/fieldset/input').click()
    driver.implicitly_wait(3)
    time.sleep(1)
    
    return driver

def get_naver_cafe_article(driver, cafe_id, article_num):
    
    cafe_article_url = 'https://cafe.naver.com/ca-fe/cafes/{cafe_id}/articles/{article_num}'.format(
        cafe_id=str(cafe_id),
        article_num=str(article_num)
    )
    
    board = title = nick_name = date = contents = ""
    
    driver.get(cafe_article_url)
    driver.implicitly_wait(3)
    time.sleep(1)
    try:
        board = driver.find_element_by_class_name('link_board').text
    except:
        pass
    try:
        title = driver.find_element_by_class_name('title_text').text
    except:
        pass
    try:
        date = driver.find_element_by_class_name('date').text
    except:
        pass
    try:
        nick_name = driver.find_element_by_class_name('nickname').text
        contents = driver.find_element_by_class_name('ContentRenderer').text
    except:
        raise Exception('Failed to parse nick name or contents.')

    article_dict = {
        'board': board,
        'title': title,
        'nick_name': nick_name,
        'date': date,
        'contents': contents
    }
    
    return article_dict
  
def main():

    if not os.path.exists(COLLECTION_DAT_DIR):
        os.makedirs(COLLECTION_DAT_DIR, exist_ok=True)

    driver = get_driver()

    naver_config = NAVER_CONFIG['main']
    naver_id = naver_config['naver_id']
    naver_passwd = naver_config['naver_passwd']

    driver = naver_login(driver, naver_id, naver_passwd)
    
    cafe_id = 11856775
    article_max_num = 386927
    result_tmp = []
    
    for article_num in range(1, article_max_num):
        try:
            article_dict = get_naver_cafe_article(driver, cafe_id, article_num)
            result_tmp.append(article_dict)
            
            split_num = 10000
            if article_num % split_num == 0:
                result = pd.DataFrame(result_tmp)
                output_path = os.path.join(COLLECTION_DAT_DIR, 'mbti_{}'.format(article_num // split_num))
                result.to_csv(output_path, index=False)
                result_tmp = []
            print('Succeed to get article({}). {}'.format(article_num, article_dict), flush=True)
        except Exception as e:
            print('Failed to get article({}). {}'.format(article_num, str(e)), flush=True)
            continue
        

if __name__ == '__main__':

    main()