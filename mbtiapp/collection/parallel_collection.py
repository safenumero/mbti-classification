# -*- coding: utf-8 -*-
# pip install tendo
# pip install pyarrow
# nohup python /Users/a/mbti/mbtiapp/collection/parallel_collection.py > /Users/a/mbti/mbtidat/collection/log/parallel_collection.log 2>&1 &

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Value, Lock, Manager, Pool
from itertools import product
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
from pathlib import Path
from tendo import singleton

common_func_dir = os.path.abspath((Path(__file__).parent / '../../common_func').resolve())
sys.path.append(common_func_dir)

from config import NAVER_CONFIG, COLLECTION_DAT_DIR, COLLECTION_OUT_DIR
from logger import JsonLogger

logging = JsonLogger().get_logger()

def export_output_to_parquet(file_id, collection_dat_dir, data_dict_list):

    if not os.path.exists(collection_dat_dir):
        os.makedirs(collection_dat_dir, exist_ok=True)
    output_path = os.path.join(collection_dat_dir, '{}.parquet'.format(file_id))

    retried = 0; max_attempts = 5; err_msg = []
    while (retried < max_attempts):
        try:

            df = pd.DataFrame(data_dict_list)
            df.to_parquet(output_path, engine='pyarrow', index=False)
            message = {
                "output_path": output_path,
                "to_parquet_status": "SUCCESS"
            }
            logging.info(message)
            break
        except Exception as e:
            retried += 1
            err_msg.append(str(e))
            time.sleep(1)
    if retried >= max_attempts:
        raise Exception('to_parquet({}) is failed. {}'.format(output_path, ', '.join(err_msg)))


def get_naver_cafe_article(p_p_article_num_list, p_driver, p_cafe_id, p_p_result_dict_list):

    split_num = 1000
    pid = os.getpid()
    logging.info(pid)
	
    result_tmp_dict_list = []
    for count, article_num in enumerate(p_p_article_num_list):
        try:
            cafe_article_url = 'https://cafe.naver.com/ca-fe/cafes/{cafe_id}/articles/{article_num}'.format(
                cafe_id=str(p_cafe_id),
                article_num=str(article_num)
            )
            
            board = title = nick_name = date = contents = ""
            
            p_driver.get(cafe_article_url)
            p_driver.implicitly_wait(1)
            time.sleep(1)
            try:
                board = p_driver.find_element_by_class_name('link_board').text
            except:
                pass
            try:
                title = p_driver.find_element_by_class_name('title_text').text
            except:
                pass
            try:
                date = p_driver.find_element_by_class_name('date').text
            except:
                pass
            try:
                nick_name = p_driver.find_element_by_class_name('nickname').text
                contents = p_driver.find_element_by_class_name('ContentRenderer').text
            except:
                raise Exception('Failed to parse nick name or contents.')

            article_dict = {
                'article_num': article_num,
                'date': date,
                'board': board,
                'title': title,
                'nick_name': nick_name,
                'contents': contents
            }

            p_p_result_dict_list.append(article_dict)
            
            result_tmp_dict_list.append(article_dict)
            
            logging.info('Succeed to get article({}, {}). {}'.format(pid, article_num, article_dict))
        except Exception as e:
            logging.error('Failed to get article({}, {}). {}'.format(pid, article_num, str(e)))

        try:
            if (count + 1) % split_num == 0:
                file_id = 'mbti_{}_{}'.format(pid, str((count + 1) // split_num).rjust(2, '0'))
                export_output_to_parquet(file_id, COLLECTION_DAT_DIR, result_tmp_dict_list)
                result_tmp_dict_list = []
            elif count + 1 >= len(p_p_article_num_list):
                file_id = 'mbti_{}_{}'.format(pid, str(((count + 1) // split_num) + 1).rjust(2, '0'))
                export_output_to_parquet(file_id, COLLECTION_DAT_DIR, result_tmp_dict_list)
        except Exception as e:
            logging.error(e)


def naver_login(p_naver_id, p_naver_passwd):
    
    retried = 0; max_attempts = 20
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
            driver.implicitly_wait(60)
            time.sleep(2)
            login_url = 'https://nid.naver.com/nidlogin.login?svctype=262144&url=http://m.naver.com/'
            driver.get(login_url)
            driver.implicitly_wait(60)
            time.sleep(2)
            driver.execute_script("document.getElementsByName('id')[0].value=\'" + p_naver_id + "\'")
            time.sleep(2) 
            driver.execute_script("document.getElementsByName('pw')[0].value=\'" + p_naver_passwd + "\'")
            time.sleep(2) 
            driver.find_element_by_xpath('//*[@id="frmNIDLogin"]/fieldset/input').click()
            driver.implicitly_wait(60)
            time.sleep(2)
            break
        except:
            driver.quit()
            retried += 1
            time.sleep(1)

    if retried >= max_attempts:
        raise Exception('Failed to get chrome driver.')

    return driver

def do_parallel(p_article_num_list, p_result_dict_list):

    naver_id = NAVER_CONFIG['sub']['naver_id']
    naver_passwd = NAVER_CONFIG['sub']['naver_passwd']
    cafe_id = 11856775

    driver = naver_login(naver_id, naver_passwd)
    try:
        get_naver_cafe_article(p_article_num_list, driver, cafe_id, p_result_dict_list)
    finally:
        driver.quit()

def main():

    start_time = datetime.now()

    message = {
        'start_time': str(start_time),
        'collection_status': 'IN_PROGRESS'
    }
    logging.info(message)

    try:

        if not os.path.exists(COLLECTION_DAT_DIR):
            os.makedirs(COLLECTION_DAT_DIR, exist_ok=True)

        manager = Manager()
        result_dict_list = manager.list()

        article_max_num = 387000
        article_num_list = range(1, article_max_num)

        # df = pd.read_csv(os.path.join(COLLECTION_OUT_DIR, 'failure_id_df.csv'))
        # article_num_list = list(df['article_num'])

        cpu_count = min(int(mp.cpu_count()), len(article_num_list))
        iterable = product(np.array_split(article_num_list, cpu_count), [result_dict_list])

        try:
            with Pool(processes=cpu_count) as pool:
                pool.starmap(do_parallel, iterable)
        except Exception as e:
            raise Exception(e)

        export_output_to_parquet('mbti_df', COLLECTION_DAT_DIR, list(result_dict_list))

        end_time = datetime.now()
        duration = end_time - start_time
        message = {
            'end_time': str(end_time),
            'duration': str(duration),
            'collection_status': 'SUCCESS'
        }
        logging.info(message)

    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        message = {
            'end_time': str(end_time),
            'duration': str(duration),
            'collection_status_desc': str(e),
            'collection_status': 'FAILURE'
        }
        logging.error(message)
        raise Exception(message)
    

if __name__ == '__main__':

    try:
        me = singleton.SingleInstance()
    except:
        sys.exit(-1)

    main()