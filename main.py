#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

# ipt = {'-R':radius, '-q':r, '-a':a, '-s':smh, '-p':p}
# R:4 radius = [20, 50, 80, 110]
# s:4 smh = [(2,3), (4,6), (6,9), (8,12)]
# p:4 p = [0, 20, 60, 100]
# q:4 r = [0, 0.02, 0.05, 0.1]
# a:4 a = [0, 0.3, 0.6, 1.0]
folder = 'data-30'

repeat_times = 50
if not os.path.exists(folder):
    os.makedirs(folder)

arg = 'p'
cmd = 'py simulate.py -t {} -f {} -a {}'.format(repeat_times, folder, arg) 
os.system('nohup unbuffer {} > {}/out.log &'.format(cmd, folder))
print(cmd)
print('done.')
