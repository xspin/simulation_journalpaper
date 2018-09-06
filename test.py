import numpy as np
import matplotlib.pyplot as plt
import time, datetime, os
from numpy.linalg import norm
# np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})

def PER(sinr, n=2):
    a = [274.7229, 90.2514, 67.6181, 53.3987, 35.3508]
    g = [7.9932, 3.4998, 1.6883, 0.3756, 0.0900]
    an, gn = a[n], g[n]
    # pn = np.exp(an)/gn 
    return np.vectorize(min)(an*np.exp(-gn*sinr), 1.0) #if sinr>pn else 1

def PER_grad(sinr, n=2):
    a = [274.7229, 90.2514, 67.6181, 53.3987, 35.3508]
    g = [7.9932, 3.4998, 1.6883, 0.3756, 0.0900]
    an, gn = a[n], g[n]
    pn = np.log(an)/gn
    print(pn)
    if sinr>pn: return -an*gn*np.exp(-gn*sinr)
    return 0

a = [0,1,2,2,2,1]
a = np.array(a)
a = a[:5:1]

def plot1():
    plt.figure()
    sinr = np.linspace(-10,20,100)
    for n in range(5):
        per = PER(sinr, n)
        plt.plot(sinr, per, label='%d'%n)
    plt.xlabel('SINR (dB)')
    plt.ylabel('PER')
    plt.title('PER vs SINR')
    plt.ylim([0, 1.1])
    plt.legend(loc=4)
    plt.savefig('PER vs SINR')

def plot2():
    d = np.linspace(1, 10, 100)
    y = 1-np.exp(-d/np.e)
    plt.figure
    plt.plot(d, y, lw=2)
    plt.ylabel('PER Upper Bound')
    plt.xlabel('$\Delta/T$')
    plt.savefig('per.png')


    plt.figure()
    plt.hold(True)

    for d in range(2,6):
        k = np.linspace(0.5, d, 50)
        y=1-(k/d)**k
        plt.plot(k, y, label='$\Delta/T={}$, $k^*={:.2f}$'.format(d, d/np.e), lw=2)
        plt.ylabel('PER Upper Bound')
        plt.xlabel('$k$')
    plt.legend()
    plt.savefig('per1.png')


def plot3():
    # x = np.linspace(-5, 40, 100)
    d = np.linspace(1, 200, 100)
    plt.figure()
    # z = np.power(10, x/10)

    # plt.plot(x, z, label='Power ratio')
    x = 10*np.log10(100/d**2/0.01)
    for i in range(5):
        y = PER(x, i)
        plt.plot(d, y, label='%d'%i)
    plt.legend(loc=4)
    plt.ylim([0, 1.2])
    plt.ylabel('PER')
    plt.xlabel('distance (m)')
    plt.title('PER vs Dist')
    plt.savefig('PER vs Dist')

    plt.figure()
    plt.plot(d,x)
    plt.ylabel('SINR (dB)')
    plt.xlabel('distance (m)')
    plt.title('SINR vs Dist')
    plt.savefig('SINR vs Dist')

def plot4():
    # x = np.linspace(-5, 40, 100)
    d = np.linspace(1, 200, 100)
    plt.figure()
    # z = np.power(10, x/10)

    # plt.plot(x, z, label='Power ratio')
    x = 10*np.log10(100/d**2/0.01)
    for i in range(5):
        y = 1/(1-PER(x, i))
        plt.plot(d, y, label='%d'%i)
    plt.legend(loc=4)
    plt.ylim([0, 12])
    plt.ylabel('Delay (s)')
    plt.xlabel('distance (m)')
    plt.title('Delay vs Dist')
    plt.savefig('Delay vs Dist')

def plot5():
    sinr = np.linspace(-1,80, 100)
    per = PER(sinr)
    pergrad = list(map(PER_grad, sinr))
    plt.figure()
    plt.plot(sinr, per, 'b-', label='PER')
    plt.plot(sinr, pergrad, 'r-', label='Grad')
    plt.xlabel('SINR')
    plt.show()


def constraint(x, lb=None, ub=None):
    if not lb is None:
        x = np.vectorize(max)(x, lb)
    if not ub is None:
        x = np.vectorize(min)(x, ub)
    return x

def plottest():

    t= np.arange(1000)/100.
    x = np.sin(2*np.pi*10*t)
    y = np.cos(2*np.pi*10*t)

    fig=plt.figure(figsize=(8,4))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(t,x)
    ax2.plot(t,y)

    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])
    # ax2.autoscale() ## call autoscale if needed

    plt.show()

def difftime(ta, tb):
    pass
# plottest()
# plot1()
# plot5()
# x = np.linspace(0,100,1000)
# plt.figure()
# plt.plot(x, 1-1/(1+x))
# plt.plot(x, 1/(1+x)**2)

# plt.show()

def sec2date(sec):
    sec = int(sec)
    return '{}:{}:{}'.format(sec//3600, sec//60%60, sec%60)

def myprint(s, color, bold=False):
    if color in ('red', 'r'):
        s = (bcolor['FAIL']+s+bcolor['ENDC'])
    elif color in ('blue', 'b'):
        s = (bcolor['OKBLUE']+s+bcolor['ENDC'])
    elif color in ('green', 'g'):
        s = (bcolor['OKBLUE']+s+bcolor['ENDC'])
    elif color in ('yellow', 'y'):
        s = (bcolor['WARNING']+s+bcolor['ENDC'])
    elif color in ('bold'):
        s = (bcolor['BOLD']+s+bcolor['ENDC'])
    if bold:
        s = (bcolor['BOLD']+s+bcolor['ENDC'])
    print(s)


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

# for key, v in bcolor.items():
#     print(v + key + bcolor['ENDC'])

# myprint('aksdkfjsab', 'r', bold=True)


import psutil
import subprocess
 
EXPAND = 1024 * 1024
 
def mems():
    ''' 获取系统内存使用情况 '''
    mem = psutil.virtual_memory()
    mem_str = " 内存状态如下:\n"
    mem_str += "   系统的内存容量为: " + str(mem.total / EXPAND) + " MB\n"
    mem_str += "   系统的内存已使用容量为: " + str(mem.used / EXPAND) + " MB\n"
    mem_str += "   系统可用的内存容量为: " + str(mem.total / EXPAND - mem.used / (1024 * 1024)) + " MB\n"
    # mem_str += "   内存的buffer容量为: " + str(mem.buffers / EXPAND) + " MB\n"
    # mem_str += "   内存的cache容量为:" + str(mem.cached / EXPAND) + " MB\n"
    return mem_str

def memused():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    memtotal = mem.free+mem.used+mem.shared
    memtotal = mem.total
    total = memtotal+swap.total
    used = mem.used+swap.used
    used = mem.used/total*100
    return '{:.2f}%'.format(used)

# print(os.path.join('asdf/','sd'))

def getpro(p='chrome'):
    s = (subprocess.check_output('pidof %s'%p, shell=True))
    s = str(s)
    s.split()
    print(s)
    pid = ''
    return pid

# p = getpro('wall')
# print(p)
# s = ','.join(map(str,[1,2,3]))
# print(s)
import fcntl  
  
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
  
# lock = Lock("/tmp/lock_name.tmp")  
# fh = lock.acquire()  
# fh.write('hello\n')
# lock.release() 

import threading, random

#生产者类
class Producer(threading.Thread):
    def __init__(self, name,sem):
        threading.Thread.__init__(self, name=name)
        self.sem=sem

    def run(self):
        for i in range(5):
            print("%s is producing %d to the queue!" % (self.getName(), i))
            self.sem.release()
            time.sleep(random.randrange(10)/5)
        print("%s finished!" % self.getName())

#消费者类
class Consumer(threading.Thread):
    def __init__(self,name,sem):
        threading.Thread.__init__(self,name=name)
        self.sem=sem
    def run(self):
        for i in range(5):
            self.sem.acquire()
            print("%s is consuming. %d is consumed!" % (self.getName(),i))
        print("%s finished!" % self.getName())

def main():

    sem = threading.Semaphore(0)
    producer = Producer('Producer',sem)
    consumer = Consumer('Consumer',sem)

    producer.start()
    consumer.start()

    print('join start ')
    producer.join()
    print('join midle ')
    consumer.join()
    print('join end ')
    print ('All threads finished!')
    print ('All threads finished! 2')
    print ('All threads finished! 3')

# main()

a = list(range(10))
print(a[0::2])
print(a[1::2])