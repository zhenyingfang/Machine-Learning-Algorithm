'''
@Author: fzy
@Date: 2019-05-23 14:54:38
@LastEditors: Zhenying
@LastEditTime: 2019-05-23 14:56:18
@Description: 
'''
import logging

def get_logger(algorithm_name):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(algorithm_name + '.log')
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
