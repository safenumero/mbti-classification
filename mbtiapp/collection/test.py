import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Value, Lock, Manager, Pool
from itertools import product
import os

def do_parallel(p_article_num_list, p_result_dict_list):
	split_num = 100
	for count, article_num in enumerate(p_article_num_list):
		message = {
			"result": int(article_num) + 100000
		}
		p_result_dict_list.append(message)


		if (count + 1) % split_num == 0:
			file_id = 'mbti_{}_{}'.format(os.getpid(), str((count + 1) // split_num).rjust(2, '0'))
			print(count, file_id)
		elif count + 1 >= len(p_article_num_list):
			file_id = 'mbti_{}_{}'.format(os.getpid(), str(((count + 1) // split_num) + 1).rjust(2, '0'))
			print('last')
			print(count, file_id)

if __name__ == '__main__':
	try:
		try:
			print('a')
			raise
		finally:
			print('b')
	except Exception as e:
		print(e)
	# manager = Manager()
	# result_dict_list = manager.list()

	# article_max_num = 387000
	# article_num_list = range(1, article_max_num)
	# cpu_count = min(int(mp.cpu_count()), len(article_num_list))
	# iterable = product(np.array_split(article_num_list, cpu_count), [result_dict_list])

	# split_num = 1000

	# for test_list in np.array_split(article_num_list, cpu_count):
	# 	result = []
	# 	for count, test in enumerate(test_list):
	# 		result.append(test)
	# 		if (count + 1) % split_num == 0:
	# 			result = []
	# 		elif count + 1 >= len(test_list):
	# 			print(result)


