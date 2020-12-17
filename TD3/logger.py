import time
import logging
import os
import sys

main_dir = os.path.abspath(os.path.dirname(__file__))
log_dir = os.path.join(main_dir, 'log')
sub_log_dir = os.path.join(log_dir, sys.argv[0].split('.')[0])
os.makedirs(sub_log_dir, exist_ok=True)
log_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
log_name1 = 'latest'
file_name = os.path.join(sub_log_dir, log_name + '.log')
file_name1 = os.path.join(sub_log_dir, log_name1 + '.log')
if os.path.exists(file_name1):
    try:
        os.remove(file_name1)
    except:
        pass

logger = logging.getLogger('multi_agent')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(file_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

fh1 = logging.FileHandler(file_name1)
fh1.setLevel(logging.DEBUG)
fh1.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(fh1)
logger.addHandler(ch)

