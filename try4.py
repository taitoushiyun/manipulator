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


    # print(type(t))
    # for key in t.keys():
    #     print(key)
    # print(type(t['jsons']))
    # print(type(t['reload']))
    # print('-'*20)
    # for key in t['jsons'].keys():
    #     print(key)
    # print("-"*40)
    # for key in t['reload'].keys():
    #     print(key)
    # print(type(t['jsons']['result']))
    # for key, value in t['jsons']['result'].items():
    #     print(key, value)
    # for key, value in t['jsons']['result']['content'].items():
    #     print(key, value)
    # for key, value in t['jsons']['eval success rate']['content']['data'][0].items():
    #     print(key, value)
    # for key, value in t['reload']['eval success rate'].items():
    #     print(key, value)
    # print(len(t['jsons']['result']['content']))
    # show(t, '')
    # print(len(t['jsons']['epoch208']['content']['data'][0]))
    # for key, value in t['jsons']['epoch208']['content'].items():
    #     print(key, value)
    # print(type(t['jsons']['epoch208']['content']['data'][0]))
    # for i in range(len(t['jsons']['epoch208']['content']['data'][0])):
    #     print(t['jsons']['epoch208']['content']['data'][0][i])
    # for key, value in t['jsons']['epoch208']['content']['data'][0].items():
    #     print(key, value)
    # print(type(t['jsons']['epoch208']['content']['data'][0]['z']))

