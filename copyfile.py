#!/usr/bin/python3
# -*- coding: utf-8 -*-

# import shutil as st
import os

ssh = True
filelist = ['simulate.py', 'summary.py', 'simrun.py', 'sim', 'main.py']
# filelist = filelist[:1]
# filelist = [filelist[2]]
# filelist = ['plot.py']

def push():
    if not ssh:
        dst = '/run/user/1000/gvfs/smb-share:server=192.168.2.2,share=f$/simulation'

        for fn in filelist:
            print('Copy:', fn)
            # st.copy2(fn, dst)
            os.system('cp -r {} {}'.format(fn, dst))
    else:
        for fn in filelist:
            print('scp Copy:', fn)
        os.system('scp -P 21215 -r {} {}'.format(' '.join(filelist), 'cx@210.28.133.11:~/simulation'))
            # os.system('cp -r {} {}'.format(fn, dst))
def pull(datapath):
    if ssh:
        print('scp pull from:', datapath)
        os.system('scp -P 21215 -r {} {}'.format('cx@210.28.133.11:~/simulation/'+datapath, './'))




pull('data-30')
# push()
print('Done.')
