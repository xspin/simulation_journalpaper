#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time, os
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})

# ipt = {'-R':radius, '-q':r, '-a':a, '-s':smh, '-p':p}
arg = '-s'

def value(s):
    s = s.split('-')
    # print(s)
    return list(map(float, s))

home = r'/run/user/1000/gvfs/smb-share:server=192.168.2.2,share=f$/simulation'
path = home+'/test'+arg
path = 'data-final/'+arg

paths = [os.path.join(path, f) for f in os.listdir(path) if f.isdigit() and os.path.isdir(os.path.join(path,f))]
paths = sorted(paths)

logfile = path+'/summary.txt'

print('Summarize...')
data = {}
with open(logfile, 'w') as summary:
    for p in paths:
        print(' Summarizing', p)
        summary.write(p+'\n')
        sublogs = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        sublogs = sorted(sublogs, key=lambda x:value(x[3:-4]))
        # print(sublogs)
        # exit()
        tp = sublogs[0].find('-')
        nn = p[-3:]
        for sublog in sublogs:
            print('    ', sublog)
            summary.write('  '+sublog+'\n')
            # temp = np.zeros(29)
            temps = []
            dcnt = 0
            with open(os.path.join(p, sublog)) as log:
                k = 0
                for r, line in enumerate(log.readlines()):
                    # if (r+1)%2 == 0:
                    if 1:
                        # print(r, line)
                        J = float(line[:line.find(',')])
                        if False:# and J<0 or J>10000:
                            # print('Delete a record:', J)
                            dcnt += 1
                            continue
                        k += 1
                        summary.write('       '+line)
                        temp = list(map(float, line.split(', ')))
                        temps.append(temp)
                        # print(temp)
            # temp /= k
            avgs = np.mean(temps,0)
            stds = np.std(temps,0)
            print('\t delete', dcnt, '/', dcnt+k)
            avgstr = ', '.join(map('{:.3f}'.format, avgs))
            stdstr = ', '.join(map('{:.3f}'.format, stds))
            summary.write('  Avg: '+avgstr+'\n')
            summary.write('  Std: '+stdstr+'\n')
            key = sublog[tp+1:-4]
            if key in data: data[key].append((nn, avgstr, stdstr))
            else: data[key] = [(nn, avgstr, stdstr)]
            # data[key] = 
            # print(temp)
        summary.write('\n')

    summary.write('='*30 + '\n')
    # for key in sorted(data.keys(), key=lambda x:float(x[:x.find('-')])):
    for key in sorted(data.keys(), key=value):
        print('  write', key)
        summary.write('\n'+key+'\n')
        for n, avg, std in sorted(data[key]):
            summary.write('{}: {}\n'.format(n, avg))
        for n, avg, std in sorted(data[key]):
            summary.write('{}: {}\n'.format(n, std))
print('Finised.')
