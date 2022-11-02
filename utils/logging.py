import logging

logging.basicConfig(format='%(asctime)s -%(levelname)-8s [%(lineno)d :%(filename)-20s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

def get_logger(name):
    return logging.getLogger(name)