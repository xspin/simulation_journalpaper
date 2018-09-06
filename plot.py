#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time, os
import matplotlib.pyplot as plt
import matplotlib as mp
# mp.rcParams.update({'font.size': 15})

# SMALL_SIZE = 12 

labelsize = 11
legendsize = 11
ticksize = 10

# mp.rc('font', size=SMALL_SIZE)          # controls default text sizes
# mp.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
mp.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
mp.rc('xtick', labelsize=ticksize)    # fontsize of the tick labels
mp.rc('ytick', labelsize=ticksize)    # fontsize of the tick labels
mp.rc('legend', fontsize=legendsize)    # legend fontsize
# mp.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
marker = ['o', '^', 's', 'p', 'x', '<', '^']
color = ['r', 'b', 'g', 'c', 'b', 'g', 'c']

winflag = False

home = '.'
if winflag:
    home = r'/run/user/1000/gvfs/smb-share:server=192.168.2.2,share=f$/simulation'
# xp = '.'
# figpath = os.path.join(path, 'fig')
figpath = '/home/i/Documents/Acdemics/Study/notes/Clustering/Journal/fig'

nodes = [10, 20, 30, 40, 50]
radius = [20, 50, 80, 110]
smh = [(2,3), (4,6), (6,9), (8,12)]
p = [0, 30, 60, 90]
# p = [0, 0.3, 0.6,0.9, 30, 60, 90]
r = [0.01, 0.02, 0.05, 0.1]
a = [0, 0.3, 0.6, 1.0]

# nodes = nodes[:3]

ipt = {'-R':radius, '-q':r, '-a':a, '-s':smh, '-p':p}

arg = '-R'
param = ipt[arg] 
# arg = '-p-bk'

home = 'data-final'

path = home+'/'+arg
logfile = path+'/summary.txt'
datapath = figpath+'/'+arg[1:]


tag = arg[1:] 

m = len(nodes)
data = [] 
with open(logfile) as summary:
    flag = False
    r = 0
    for line in summary.readlines():
        if not flag and line[:3] == '===':
            flag = True
            r = 0
            continue
        if flag:
            line = line.strip()
            if line=='': continue
            # if r==1:
            #     pval.append(line.split('-')[k])
            # if r > 1:
            # print('[%s]'%line)
            if line[3]==':':
                # print(line[4:])
                data[-1].append(list(map(float, line[4:].strip().split(', '))))
            else:
                if len(data)>0:
                    print(len(data[-1]), 'x',len(data[-1][0]))
                data.append([])
data = np.array(data)
if len(data.shape)<2:
    # print(data)
    print('Data Process Error!')
    exit()
# print(data.shape)
# print(pval)
if not os.path.exists(figpath):
    os.makedirs(figpath)

if not os.path.exists(datapath):
    os.makedirs(datapath)

# keys = ['F', 'P', 'S', 'W', 'NH', 'NH1', 'N1', 'N2', 'N3', 'N123', 
keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH',
        'a_avg','a_var','a_min','a_max',
        'h_avg','h_var','h_min','h_max',
        'm_avg','m_var','m_min','m_max',
        'p_avg','p_var','p_min','p_max'
        ]
# keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']
idx = {key:i for i, key in enumerate(keys)}
marker = ['s', '>', 'o', 'p', 'x', '<', '^']
loc = ['upper right', 'upper left', 'upper left', 'upper left']
ylb = {'J': 'Obj', 'F': 'Coverage Efficiency', 
        'S':'Coverage Area Efficiency', 
        'W':'Coverage Width Efficiency', 
        'P':'Average Transmission Power', 
        'D': 'Penalty', 
        'N1': '# D1',
        'N2': '# D2',
        'N3': '# D3',
        'N123': '# D',
        'NH1': '# NCH',
        'NH': 'Number of CHs',
        'a_avg':'Mean','a_var':'Variance','a_min':'Min','a_max':'Max',
        'h_avg':'Mean','h_var':'Variance','h_min':'Min','h_max':'Max',
        'm_avg':'Mean','m_var':'Variance','m_min':'Min','m_max':'Max',
        'p_avg':'Mean','p_var':'Variance','p_min':'Min','p_max':'Max',
        }

lgloc = {'F':3, 'P':4, 'S':3, 'W':1, 'NH':2, 'NH1':2, 'N1':2, 'N2':2, 'N3':2, 'N13':2}
plotkeys = ['F', 'P', 'S', 'W', 'NH', 'NH1', 'N1', 'N2', 'N3', 'N123', 'D1', 'D2', 'D3',
            'a_avg','a_var',
            'h_avg','h_var',
            'm_avg','m_var',
            'p_avg','p_var'
            ]
for key in (plotkeys):
    name = datapath+'/{}-{}'.format(tag,key)
    print('saving data to', name+'.csv')
    fp = open(name+'.csv','w')
    print('Plot', key)
    plt.figure(figsize=(4,3))
    xdata = np.array(nodes)
    ax = plt.gca()
    # ax.set_fontsize(12)
    # if key=='S' or key=='W':
    #     plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # ax.yaxis.major.formatter._useMathText = True
        # plt.ylim(0,1)
    fp.write(','.join(map(str, xdata))+'\n')
    N = m
    ymax = 0
    for i in range(len(param)):
        # print('  :', param[i])
        if key=='N123':
            ydata = data[i][:N, idx['N1']]+data[i][:N, idx['N2']]+data[i][:N, idx['N3']]
            zdata = np.max(data[i][N:, [idx['N1'], idx['N2'], idx['N3']]],1)
        else:
            ydata = data[i][:N, idx[key]]
            zdata = data[i][N:, idx[key]]
        # fp.write('{}{}\n'.format(i, param[i]))
        fp.write(','.join(map(str, ydata))+'\n')
        fp.write(','.join(map(str, zdata))+'\n')
        # ax = plt.plot(xdata, ydata, marker=marker[i],ms=6, label='{}={}'.format(tag,param[i]))
        # print(xdata.shape, ydata.shape, zdata.shape)
        lb = label='{}={}'.format(tag,param[i])
        plt.plot(xdata+(-3+i*2), ydata, marker=marker[i], color=color[i], ms=3, label=lb)
        plt.bar(xdata + (-3+i*2), ydata, yerr=zdata, align='center', alpha=0.7, width=2, ecolor='gray',edgecolor='w', color=color[i], label=lb)
        ymax = max(ymax, np.max(zdata+ydata))
    plt.xlabel('Number of Nodes')
    lb = ylb[key] if key in ylb else key 
    plt.ylabel(lb)
    if key in lgloc:
        lc = lgloc[key]
    else:
        lc = 0
    # plt.legend(loc=lc)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(xdata[0]-5, xdata[-1]+5)
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.grid(True)
    plt.rc('grid', linestyle=":", color='gray')
    plt.savefig(name+'.eps', bbox_inches='tight')
    fp.close()
    # break
         
# plt.show()
print('End.')