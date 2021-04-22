import json

def show(d, prefix):
    if isinstance(d, dict):
        for key, value in d.items():
            print(prefix + key)
            show(value, prefix + '\t')
    else:
        print(prefix, d)

with open('/home/cq/.visdom/her_17.json') as f:
    t = json.load(f)


if __name__ == '__main__':
    import time
    print('1')
    time.sleep(5)
    print('1-')
