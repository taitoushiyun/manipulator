import re
import visdom
from _collections import deque
file_name = 'test_data.txt'
with open(file_name) as f:
    lines = f.readlines()
episodes = []
results = []
path_lens = []
eval_success_queue = deque(maxlen=20)
eval_success_rate = []
for line in lines:
    if 'episode' in line and 'result' in line:
        episode, result, path_len = re.findall(r'episode (\d+) result (\d+) path len (\d+)', line)[0]
        episode = float(episode)
        result = float(result)
        path_len = float(path_len)
        eval_success_queue.append(result)
        eval_success_rate.append(sum(eval_success_queue) / len(eval_success_queue))
        episodes.append(episode)
        results.append(result)
        path_lens.append(path_len)

vis = visdom.Visdom(port=6016, env='td3_23')
# vis.line(X=[0], Y=[0], win='result', opts=dict(Xlabel='episode', Ylabel='result', title='result'))
vis.line(X=[0], Y=[0], win='eval result', opts=dict(Xlabel='episode', Ylabel='eval result', title='eval result'))
vis.line(X=[0], Y=[0], win='eval path len', opts=dict(Xlabel='episode', Ylabel='len', title='eval path len'))
vis.line(X=[0], Y=[0], win='eval success rate', opts=dict(Xlabel='episode', Ylabel='success rate (%)', title='eval success rate'))

vis.line(X=episodes, Y=results, win='eval result', update='append')
vis.line(X=episodes, Y=path_lens, win='eval path len', update='append')
vis.line(X=episodes, Y=[100*i for i in eval_success_rate], win='eval success rate', update='append')

