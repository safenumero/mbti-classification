import os
import json
from pathlib import Path

ROOT_DIR = os.path.abspath(Path(__file__).parent / '../../mbti')
APP_DIR = os.path.join(ROOT_DIR, 'mbtiapp')
DAT_DIR = os.path.join(ROOT_DIR, 'mbtidat')
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')

try:
    NAVER_CONFIG = json.load(open(os.path.join(CONFIG_DIR, 'naver_config.json'), "r"))
except:
    NAVER_CONFIG = {
        'sub': {
            'naver_id': 'test',
            'naver_passwd': 'test'
        }
    }

COLLECTION_DAT_DIR = os.path.join(DAT_DIR, 'collection', 'dat')
COLLECTION_OUT_DIR = os.path.join(DAT_DIR, 'collection', 'out')
PREPROC_EDA_DAT_DIR = os.path.join(DAT_DIR, 'preproc_eda', 'dat')
PREPROC_EDA_OUT_DIR = os.path.join(DAT_DIR, 'preproc_eda', 'out')
MODELING_DAT_DIR = os.path.join(DAT_DIR, 'modeling', 'dat')
MODELING_OUT_DIR = os.path.join(DAT_DIR, 'modeling', 'out')
