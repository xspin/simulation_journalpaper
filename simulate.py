#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time, os, datetime, threading, getopt, sys
import sim
import sim.bcd
import sim.graph
import sim.calc
import matplotlib.pyplot as plt
import psutil

np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})

def clear(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            exit()

def sec2date(sec):
    sec = int(sec)
    return '{}:{}:{}'.format(sec//3600, sec//60%60, sec%60)

def monitor(asksem, cpusem, cnt):
    k = 0
    while True:
        asksem.acquire()
        if exit_flag:
            break
        cpu = psutil.cpu_percent(interval=1)/100*32
        myprint('###CPU used: {}'.format(cpu), 'r')
        if cpu<30.6:
            k += 1
            cpusem.release()
            myprint('###[start a new thread] {}/{}'.format(k, cnt), 'r')
            if k>=cnt: break
        time.sleep(1+np.random.randint(10))
    myprint('###Monitor exists', 'r')

# def main(sem, nodes, m, arg, v):
def main(home_path, asksem, sem, N, arg, v, k, m, t, times):
    # asksem.release()
    sem.acquire()
    global errors
    stopfile = os.path.join(folder, 'stop')
    if os.path.exists(stopfile):
        myprint('Terminated by user', 'r')

    if arg=='-s': tag = '{} ({:2d}, {:2d})'.format(arg, *v)
    else: tag = '{}{:6}'.format(arg, v)
    name = '[Thread {}]'.format(tag)
    # times = len(nodes) * m
    # t = 0
    # rep_start = time.time()
    # for k in range(m):
    #     for N in nodes:
    # sim.parameter['#nodes'] = N
    # t += 1
    msg = '{}  [N{:02d} {:02d}/{} {}/{}]'.format(name, N, t, times, k, m)
    run_start = time.time()
    if arg=='-s':
        argv = '-m {} -h {}'.format(*v)
    else:
        argv = '{} {}'.format(arg, v)
    args = '{} -N {} -g "{}" -T {} -F "{}"'.format(argv, N, msg, timestamp, home_path)
    o = os.system('python3 simrun.py {}'.format(args))
    # print(args)
    # o = 0
    # time.sleep(0.5)
    # o = 0
    if o>0:
        errors.append(args)
        # exit()
    # time.sleep(3)
    now = time.time()
    run_time = now-run_start
    myprint('{}  [k{}/{} N{:02d}] One-Run Time: {}'.format(msg, k+1, m, N, sec2date(run_time)), 'y')
    cost_time = sec2date(now-timestamp)
    myprint('{}  Total Cost Time: {}'.format(msg, cost_time), 'g')
    # rep_time = time.time() - rep_start 
    # myprint('{}  [k{}/{}] One-Repeat Time: {}'.format(name, k+1, m, sec2date(rep_time)), 'r')
    # myprint('{}  Remain: {}'.format(name, sec2date(rep_time*(m-k-1)/(k+1))), 'r', True)
    myprint(msg + '  Finished.', 'b')
    sem.release()
    # asksem.release()

exit_flag = False
def threadrun(nodes, m):
    sem = threading.BoundedSemaphore(31)
    asksem = threading.Semaphore(0)

    myprint('Creating Thread Object ...', 'b')

    # tlist = [threading.Thread(target=monitor, args=(asksem, sem, totalthreads))]
    tlist = []
    times = len(nodes)*m
    tid = 0
    for arg in params:
        param = ipt[arg] 
        home_path = folder+'/'+arg
        mkdirs(home_path, nodes)
        for v in param:
            i = 0
            for k in range(m):
                for N in nodes:
                    i+=1
                    tid += 1
                    t = threading.Thread(target=main, args=(home_path, asksem, sem, N, arg, v, tid, totalthreads, i, times))
                    tlist.append(t)
                    print('    Thread {}: [{}, {}, {}, {}]'.format(tid, arg, v, k, N))
    myprint('Starting Threads ...', 'b')
    for t in tlist: t.start()
    myprint('Waiting Finish ...', 'b')
    for t in tlist: t.join()
    asksem.release()

bcolor = {
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def myprint(s, color, bold=False):
    if color in ('red', 'r'):
        s = (bcolor['FAIL']+s+bcolor['ENDC'])
    elif color in ('blue', 'b'):
        s = (bcolor['OKBLUE']+s+bcolor['ENDC'])
    elif color in ('green', 'g'):
        s = (bcolor['OKGREEN']+s+bcolor['ENDC'])
    elif color in ('yellow', 'y'):
        s = (bcolor['WARNING']+s+bcolor['ENDC'])
    elif color in ('bold'):
        s = (bcolor['BOLD']+s+bcolor['ENDC'])
    if bold:
        s = (bcolor['BOLD']+s+bcolor['ENDC'])
    print(s)

def mkdirs(ppath, nodes):
    if not os.path.exists(ppath):
        os.makedirs(ppath)
    for n in nodes:
        path1 = ppath + '/%03d'%n
        if not os.path.exists(path1):
            os.makedirs(path1)

####################################
nodes = [10, 20, 30, 40, 50]
radius = [20, 50, 80, 110]
smh = [(2,3), (4,6), (6,9), (8,12)]
# p = ['true', 'false']
p = [0, 0.3, 0.6,0.9, 30, 60, 90]
r = [0.01, 0.02, 0.05, 0.1]
a = [0, 0.3, 0.6, 1.0]


# m = 5
maxThread = 4

# nodes = nodes[:3]
# radius = radius[:4]
# r = [0.2]

ipt = {'-R':radius, '-q':r, '-a':a, '-s':smh, '-p':p}
params = ['-s', '-p', '-q', '-a', '-R']
# params = ['-p']

def usage():
        print('Usage: [-?|-f|-t] args')
        print('\t -? help')
        print('\t -t repeat times')
        print('\t -f data path')

if __name__ == '__main__':
    try:
        argstr = '?t:x:a:f:'
        opts, args = getopt.getopt(sys.argv[1:], argstr)
        # print(opts, args)
        timestamp = time.time()
        if len(opts)<1:
            print ('Input Error!')
            usage()
            sys.exit(1)
        for opt, argin in opts:
            # print(opt, argin)
            if opt == '-?': 
                usage()
                exit(0)
            elif opt == '-t':
                m = int(argin)
            elif opt == '-f':
                folder = argin
            elif opt == '-a':
                if '-'+argin in params:
                    params = ['-'+argin]
            else:
                print('arg', opt, 'error')
                usage()
                exit()
    except getopt.GetoptError:
        print ('Input Error!')
        usage()
        sys.exit(1)

#################################

    global errors 

    start_time = time.time()
    timestamp = start_time


    ######################################################
    errors = []
    # total_cnt = len(nodes)*len(param)*m
    totalthreads = len(params)*4*m*len(nodes)


    myprint('Start Threading...', 'b')

    threadrun(nodes, m)

    cost = time.time() - start_time
    myprint('\nTotal Cost Time: {}'.format(sec2date(cost)), 'r')

    myprint('Error Count: {}/{}\n'.format(len(errors), totalthreads), 'y')

    if len(errors)>0:
        myprint('\nErrors:', 'r')
        for e in errors:
            myprint('{}'.format(e), 'r')