#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time, os
import matplotlib.pyplot as plt
import matplotlib as mp
import csv
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
# mp.rc('legend', labelspacing=0.5)
winflag = False

home = '.'
home  = '/home/i/Documents/Acdemics/Study/notes/Clustering/Journal/fig'

nodes = [10, 20, 30, 40, 50]
radius = [20, 50, 80, 110]
smh = [(2,3), (4,6), (6,9), (8,12)]
p = [0, 30, 60, 90]
# p = [0, 0.3, 0.6,0.9, 30, 60, 90]
r = [0.01, 0.02, 0.05, 0.1]
a = [0, 0.3, 0.6, 1.0]

# nodes = nodes[:3]

val = {'-R':'R', '-q':'r', '-a':'a', '-s':'s', '-p':'P'}
ipt = {'-R':radius, '-q':r, '-a':a, '-s':smh, '-p':p}

arg = '-p'
param = ipt[arg] 

datapath = home+'/'+arg[1:]
figpath = home+"/fig-{}".format(val[arg])
print(datapath, figpath)



if not os.path.exists(figpath):
    os.makedirs(figpath)

data = []

ykey = {'-R':['F', 'S', 'W', 'NH'], 
        '-q':['F','S','W', 'NH', 'N123'], 
        '-s':['F','S','W', 'NH', 'N123'], 
        '-a':['F', 'S', 'W', 'NH'], 
        '-p':['F','S','W','P', 'NH', 'N123'], 
        }
for k in ykey.keys():
    for i in 'ahmp':
        ykey[k].extend([i+'_avg', i+'_var'])

keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH',
        'a_avg','a_var','a_min','a_max',
        'h_avg','h_var','h_min','h_max',
        'm_avg','m_var','m_min','m_max',
        'p_avg','p_var','p_min','p_max'
        ]
idx = {key:i for i, key in enumerate(keys)}
marker = ['o', '^', 's', 'p', 'x', '<', '^']
color = ['r', 'b', 'g', 'c', 'b', 'g', 'c']
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
        'a_avg': 'Average Delay',
        'a_var': 'Delay from L to A',
        'h_avg': 'Delay from L to CHs',
        'h_var': 'Delay from L to CHs',
        'm_avg': 'Delay from CHs to CMs',
        'm_var': 'Delay from CHs to CMs',
        'p_avg': 'Power',
        'p_var': 'Power'
        }

lgloc = {'F':3, 'P':4, 'S':3, 'W':1, 'NH':2, 'NH1':2, 'N1':2, 'N2':2, 'N3':2, 'N13':2}

for key in ykey[arg]:
    print('Plot', ylb[key])
    # if key not in 'ahmp':
    name = datapath+'/{}-{}'.format(arg[1:],key)
    print('  get data from', name+'.csv')
    with open(name+'.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        data = list(reader)
    if 0:
        name_avg = datapath+'/{}-{}_avg.csv'.format(arg[1:],key)
        name_var = datapath+'/{}-{}_var.csv'.format(arg[1:],key)
        print('  get data from', name_avg, name_var)
        with open(name_avg) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            data = list(reader)
        with open(name_var) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            data_var = list(reader)

    plt.figure(figsize=(4,3))
    xdata = np.array(nodes)
    ax = plt.gca()
    if arg in ['-R', '-q', '-s', '-p'] and (key in 'FSW'):
        plt.ylim(0,1)
    if arg=='-p' and key=='P':
        plt.ylim(0,100)
    # if arg=='-a' and key=='S':
    #     plt.ylim(0.8,0.93)

    # if key not in 'ahmp':
    data = np.array(data)
    if 1:
        # for i,ydata in enumerate(data[1:]):
        for i in range(len(data)//2):
            ydata = data[1+i*2, :]
            zdata = data[2+i*2, :]
            tag = arg[1:]
            if tag=='s': tag = r'\sigma' 
            elif tag=='q': tag = 'r'
            elif tag=='p' and i!=0: 
                tag = r'\bar P' 
            lb = '${}={}$'.format(tag,param[i])
            if tag=='p' and i==0: 
                lb = 'Optimized'
            # ax = plt.plot(xdata, ydata, marker=marker[i], c=color[i], ms=6, label=lb)
            if 1 or key in 'FSWP':
                plt.plot(xdata, ydata, marker=marker[i], color=color[i], ms=6, label=lb)
                # plt.errorbar(xdata+(-2+i), ydata, zdata, marker=marker[i], color=color[i],ecolor='gray', ms=3, label=lb)
                # plt.xlim(xdata[0]-2, xdata[-1]+2)
            else:
                plt.plot(xdata+(-3+i*2), ydata, marker=marker[i], color=color[i], ms=3, label=lb)
                plt.bar(xdata + (-3+i*2), ydata, yerr=zdata, align='center', alpha=0.7, width=2, ecolor='gray',edgecolor='w', color=color[i])
        # plt.xlim(xdata[0], xdata[-1])
                plt.xlim(xdata[0]-5, xdata[-1]+5)
    if 0:
        for i, (ydata, zdata) in enumerate(zip(data[1:], data_var[1:])):
            tag = arg[1:]
            if tag=='s': tag = r'\sigma' 
            elif tag=='q': tag = 'r'
            elif tag=='p' and i!=0: 
                tag = r'\bar P' 
            lb = '${}={}$'.format(tag,param[i])
            if tag=='p' and i==0: 
                lb = 'Optimized'
            if arg=='-p':
                ax = plt.errorbar(xdata, ydata, zdata, marker=marker[i], c=color[i], ms=6, label=lb)
            else:
                ax = plt.bar(xdata + (-3+i*2), ydata, yerr=zdata, align='center', alpha=0.7, width=2, ecolor='gray',edgecolor='w', color=color[i], label=lb)

    plt.xlabel('Number of Nodes')
    plt.ylabel(ylb[key])
    if key in lgloc:
        lc = lgloc[key]
    else:
        lc = 0
    plt.legend(loc=lc, labelspacing=0.1)
    plt.tight_layout()
    plt.grid(True, ls="-", color='black', alpha=0.1)
    name = figpath+"/{}".format(key)
    print('  Save fig to', name)
    # plt.show()
    # break
    plt.savefig(name+'.pdf', bbox_inches='tight')
         
# plt.show()
print('End.')