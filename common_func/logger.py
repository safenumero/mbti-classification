# -*- coding: utf-8 -*-
import os
import time
import logging


class JsonLogger:
	
    def __init__(self, log_path=None, log_file=None):
        self._log_path = log_path
        self._log_file = log_file
        os.environ['TZ'] = 'Asia/Seoul'
        time.tzset()

        self.logger = logging.getLogger(__name__)

    def get_logger(self):

        if not self.logger.handlers:

            formatter = logging.Formatter("{'ascTime': '%(asctime)s', 'fileName': '%(filename)s' 'lineNo': '%(lineno)s', 'levelName': '%(levelname)s', 'message': '%(message)s'}")

            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)

            self.logger.addHandler(streamHandler)
            self.logger.setLevel(level=logging.DEBUG)
            self.logger.propagate = False

            if self._log_path != None and self._log_file != None:

                try:
                    os.makedirs(self._log_path, exist_ok=True)
                except OSError as e:
                    raise Exception("Can not make log directory...{}, {}".format(self._log_path, e))

                fileHandler = logging.FileHandler('{}/{}.log'.format(self._log_path, self._log_file))

                fileHandler.setFormatter(formatter)

                self.logger.addHandler(fileHandler)
                
            return self.logger

        else:
            return self.logger