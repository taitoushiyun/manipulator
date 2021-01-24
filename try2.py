import numpy as np
import time
import argparse
def main(args):
    print(args.code_version)
    time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_version', type=str, default='td3_26')
    args = parser.parse_args()
    main(args)