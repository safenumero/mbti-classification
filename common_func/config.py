import os
import json
from pathlib import Path

ROOT_DIR = os.path.abspath(Path(__file__).parent / '../../mbti')
APP_DIR = os.path.join(ROOT_DIR, 'mbtiapp')
DAT_DIR = os.path.join(ROOT_DIR, 'mbtidat')
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')

NAVER_CONFIG = json.load(open(os.path.join(CONFIG_DIR, 'naver_config.json'), "r"))

COLLECTION_DAT_DIR = os.path.join(DAT_DIR, 'collection', 'dat')
COLLECTION_OUT_DIR = os.path.join(DAT_DIR, 'collection', 'out')
