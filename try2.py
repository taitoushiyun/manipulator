import time
import logging
import os
import sys
from try3 import test

main_dir = os.path.abspath(os.path.dirname(__file__))
log_dir = os.path.join(main_dir, 'log')
sub_log_dir = os.path.join(log_dir, sys.argv[0].split('.')[0])
os.makedirs(sub_log_dir, exist_ok=True)
log_name = 'td3_47'
file_name = os.path.join(sub_log_dir, log_name + '.log')
if os.path.exists(file_name):
    try:
        os.remove(file_name)
    except:
        pass

logger = logging.getLogger('multi_agent')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(file_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.info('111')
test()
