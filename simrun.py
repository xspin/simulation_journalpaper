#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import time, os, datetime, getopt, sys
import sim
import sim.bcd
import sim.graph
import sim.calc
import matplotlib.pyplot as plt
import fcntl  

np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})

class Lock:   
    def __init__(self, filename):  
        self.filename = filename  
        # This will create it if it does not exist already  
        self.handle = open(filename, 'a')  
      
    # Bitwise OR fcntl.LOCK_NB if you need a non-blocking lock   
    def acquire(self):  
        fcntl.flock(self.handle, fcntl.LOCK_EX)  
        return self.handle
          
    def release(self):  
        fcntl.flock(self.handle, fcntl.LOCK_UN)  
        self.handle.close()  

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
# sim.figpath = 'testimg'

# seed = np.random.get_state()
# print('seed:', seed[1][0])

# seed = 3693469210
# np.random.seed(seed)



def check(var=None):
    print('Grad Checking ...')
    if var==None:
        sim.parameter['width'] = 100*np.sqrt(sim.parameter['#nodes'])
        var = sim.initialize()
    sim.calc.gain_grad_check(var)
    sim.calc.sinr_per_grad_check(var)
    sim.calc.width_grad_check(var)
    sim.calc.area_grad_check(var)
    sim.calc.delay_grad_check(var)
    sim.calc.penalty_grad_check(var)
    sim.calc.obj_grad_check(var)
    exit()

# clear(sim.figpath)
# os.remove('./'+sim.figpath)
# exit()

def makedir(ppath):
    # t = time.time()
    # t = '{:04.0f}'.format((t-int(t))*10000)
    t = '{:05.0f}'.format(np.random.rand()*100000)
    t = ('{0.tm_year}{0.tm_mon:02d}{0.tm_mday:02d}-{0.tm_hour:02d}.{0.tm_min:02d}.{0.tm_sec:02d}.{1}'.format(time.localtime(), t))
    path1 = '%03d'%sim.parameter['#nodes']
    path1 = os.path.join(ppath, path1)
    path2 = os.path.join(path1, t)
    print('Path:', path1, path2)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if plot_flag and not os.path.exists(path2):
        try:
            os.makedirs(path2)
        except:
            print('Error: Path', path2, 'already exists.')
            print('Error: Path', path2, 'already exists.')
            print('Error: Path', path2, 'already exists.')
            print('Error: Path', path2, 'already exists.')
            # with open(path1+)
        # exit()
    sim.figpath = path2 
    return t, path1, path2

def stat(var, mdelay, head):
    H = var['head']
    P = var['power']
    N = len(H)
    dl2a = [mdelay[0,i] for i in range(N) if mdelay[0,i]<np.inf]
    dl2h = [mdelay[0,i] for i in range(N) if H[i] and mdelay[0,i]<np.inf]
    dh2m = [mdelay[head[i],i] for i in range(N) if head[i]>=0]
    st = lambda x: [np.mean(x), np.std(x), np.min(x), np.max(x)]
    return st(dl2a), st(dl2h), st(dh2m), st(P)


def run(msg, timestamp, Path):
    stopfile = os.path.join(Path, 'stop')
    if os.path.exists(stopfile):
        print(msg, 'Terminated.')
        with open(stopfile, 'a') as fp:
            fp.write(str(datetime.datetime.now())+'\t')
            fp.write(msg+'\n')
        return False
            
    flag_log  = True
    # flag_log  = False
    # initialize
    # exit()

    sim.parameter['width'] = 100*np.sqrt(sim.parameter['#nodes'])
    # print('width:', sim.parameter['width'])
    tp = 'random'

    # t = ('{0.tm_year}{0.tm_mon:02d}{0.tm_mday:02d}'.format(time.localtime()))
    print(msg, 'Time:', time.ctime())
    tm, path1, path2 = makedir(Path)
    if flag_log:
        param = ''
        for key in ['#nodes', 'RC', 'probability', 'a', 'delay_h2m', 'delay_l2h']:
            param += ('{}-'.format(sim.parameter[key]))
        # param += tp[0].upper()
        param += str(sim.parameter['P'])
        print(msg, 'Paramerter:', param)

        # fp = open(path1+'/log'+param+'.txt', 'a')
        # fp.write(tm+'\n')
        # fname = path1+'/log'+param+'.txt'
        # lock = Lock(fname)  
        # fp = lock.acquire()  
        # fp.write(tm+'\n')
        # lock.release() 

    N = sim.parameter['#nodes']
    var = sim.initialize(type=tp, heads=False)
    # Q = var['position']
    # R = sim.parameter['RC']
    # Q[1] = [50,300]
    # print("#Nodes:", N)
    # print('Type:', tp)
    # if flag_log:
        # fp.write('# of Nodes: %d\n'%N)
        # fp.write('Type: %s\n'%tp)
        # fp.write(str(sim.parameter)+'\n')
        

    sinr, per, delay, mdelay, path = sim.graph.delay_graph(var)
    mpath, hpath, head = sim.graph.graph(var, delay, mdelay, path)
    if plot_flag:
        sim.graph.plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, title='Initial State', filename=sim.figpath+'/!%d-(%s)initial.png'%(N,tp))

    # check()

    # sinr, per, delay, mdelay, mpath, hpath, head, allvals = sim.bcd.block_coordinate_descent(var)
    sinr, per, delay, mdelay, mpath, hpath, head, allvals = sim.bcd.greedy_descent(var, msg, timestamp)
    # keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']
    allvals = np.array(allvals)
    obj_values = allvals[:, 0]
    ee_values = allvals[:, 1]
    s_values = allvals[:, 2]   
    w_values = allvals[:, 3]   
    p_values = allvals[:, 4]   
    # print(var['power'])
    st1, st2, st3, st4 = stat(var, mdelay, head)


    # keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']
    if flag_log:
        # fp.write('Obj(J): %s\n'%str(obj_values))
        # fp.write('EE(F): %s\n'%str(allvals[:,1]))
        # fp.write('Width(W): %s\n'%str(allvals[:,2]))
        # fp.write('Pt: %s\n'%str(allvals[:,3]))
        # fp.write('Pm: %s\n'%str(allvals[:,4]))
        # fp.write('d1: %s\n'%str(allvals[:,5]))
        # fp.write('d2: %s\n'%str(allvals[:,6]))
        # fp.write('d3: %s\n'%str(allvals[:,7]))
        # fp.write('Final:\n')
        # vals = ['J', 'F', 'W', 'Pt', 'Pm', 'd1', 'd2', 'd3']
        # for i,v in enumerate(vals):
        #     fp.write('  {} = {}\n'.format(v, allvals[-1,i]))
        # fp.write('Power: '+ str(sum(var['power'])) +'\n')
        # fp.write(str(var['power'])+'\n')
        vals = allvals[-1]
        vals = np.r_[vals, st1, st2, st3, st4]
        # fp.write(', '.join(map('{:.3f}'.format, vals)))
        # fp.write('\n')
        # fp.close()

        fname = path1+'/log'+param+'.txt'
        lock = Lock(fname)  
        fp = lock.acquire()  
        fp.write(', '.join(map('{:.3f}'.format, vals)))
        fp.write('\n')
        lock.release() 

    if plot_flag:
        title = 'Final State'
        sim.graph.plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=sim.figpath+'/!%d-(%s)final.png'%(N,tp), title=title)

        plt.figure(figsize=(8,12))
        f, ax = plt.subplots(5, sharex=True)

        ax[0].set_title('Evolution')
        n = len(obj_values)
        # obj_values = obj_values[0:n:max(1,n//80)]
        ax[0].plot(obj_values, '*--', label='J')
        # plt.xlabel('Iteration')
        ax[0].set_ylabel('Obj')
        a = min(obj_values)
        b = max(obj_values)
        d = max((b-a)/n, 0.00001)
        ax[0].set_ylim([a-d, b+d])

        ax[1].plot(ee_values, 's--', label='F')
        # plt.xlabel('Iteration')
        ax[1].set_ylabel('F ($m^2/mW$)')
        a = min(ee_values)
        b = max(ee_values)
        d = max((b-a)/n, 1)
        ax[1].set_ylim([a-d, b+d])

        ax[2].plot(s_values, '>--', label='S')
        # ax[].set_xlabel('Iteration')
        ax[2].set_ylabel('Area ($m^2$)')
        a = min(s_values)
        b = max(s_values)
        d = max((b-a)/n, 1)
        ax[2].set_ylim([a-d, b+d])

        ax[3].plot(w_values, '<--', label='W')
        # ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel('Width (m)')
        a = min(w_values)
        b = max(w_values)
        d = max((b-a)/n, 1)
        ax[3].set_ylim([a-d, b+d])

        ax[4].plot(p_values, '.--', label='P')
        ax[4].set_xlabel('Iteration')
        ax[4].set_ylabel('Power ($mW$)')
        a = min(p_values)
        b = max(p_values)
        d = max((b-a)/n, 1)
        ax[4].set_ylim([a-d, b+d])


        filename = sim.figpath+"/!{}-({})evolution.png".format(N, tp)
        plt.savefig(filename)
        plt.close()
    return True

def sec2date(sec):
    sec = int(sec)
    return '{}:{}:{}'.format(sec//3600, sec//60%60, sec%60)

def usage():
        print('Usage: [-?|-N|-R|-m|-h|-p|-r|-a|-g] args')
        print('\t -? help')
        print('\t -N #Nodes')
        print('\t -R Radius')
        print('\t -m Sm')
        print('\t -h Sh')
        print('\t -p Power')
        print('\t -r Interference Rate')
        print('\t -a Weight')
        print('\t -g msg')
        print('\t -T timestamp')

if __name__ == '__main__':
    try:
        argstr = '?N:R:m:h:p:q:a:g:T:F:'
        opts, args = getopt.getopt(sys.argv[1:], argstr)
        # print(opts, args)
        msg = ''
        timestamp = time.time()
        Path = 'tmp'
        for opt, arg in opts:
            # print(opt, arg)
            if opt == '-?': 
                usage()
                exit()
            elif opt == '-N':
                sim.parameter['#nodes'] = int(arg) 
            elif opt == '-R':
                sim.parameter['RC'] = float(arg) 
            elif opt == '-m':
                sim.parameter['delay_h2m'] = float(arg) 
            elif opt == '-h':
                sim.parameter['delay_l2h'] = float(arg) 
            elif opt == '-p':
                # sim.parameter['']
                sim.parameter['P'] = float(arg) 
            elif opt == '-q':
                sim.parameter['probability'] = float(arg) 
            elif opt == '-a':
                sim.parameter['a'] = float(arg) 
            elif opt == '-g':
                msg = arg
            elif opt == '-T':
                timestamp = float(arg)
            elif opt == '-F':
                Path = arg
            else:
                print('arg', opt, 'error')
                usage()
                exit()
    except getopt.GetoptError:
        print ('Input Error!')
        usage()
        sys.exit(1)
    plot_flag = False
    run(msg, timestamp, Path)
    print('done.')
 