import sys
import os

main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(main_dir)
sys.path.append(os.path.join(main_dir, 'TD3'))
sys.path.append(os.path.join(main_dir, 'vrep_pyrep'))

