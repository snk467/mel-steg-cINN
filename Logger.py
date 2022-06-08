import logging
logging.basicConfig(level = logging.INFO,format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')

def get_logger(name):
    return logging.getLogger(name)