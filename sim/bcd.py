#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
import time
import sim
from sim.graph import graph, delay_graph, plot
from sim.calc import obj, objfromvar

def A_clustering(var):
    flag = False
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    N = len(H)
    sinr, per, delay, mdelay, path = delay_graph(var)
    H[0] = True
    for k in range(max(np.ceil(N/5/2).astype(int),1)):
        # print('\titer', k)
        d = 0
        t = 0
        mpath, hpath, head = graph(var, delay, mdelay, path)
        current_obj = obj(var, mdelay, mpath, hpath, head, flag=False)
        # print('CHs:', np.array(list(range(N)))[var['head']])
        # print('curent obj:{}'.format(current_obj))
        # print('  d:', end='')
        for i in range(1,N):
            # Add CH
            if not H[i]:
                H[i] = True
                mpath, hpath, head = graph(var, delay, mdelay, path)
                temp_obj = obj(var, mdelay, mpath, hpath, head)
                if current_obj - temp_obj>d+1e-5:
                    d = current_obj - temp_obj
                    t = i
                    # print(' {}: {:.2f} = {}-{}'.format(i, current_obj-temp_obj, current_obj, temp_obj))
                H[i] = False
        if t>0: 
            H[t] = True
            mpath, hpath, head = graph(var, delay, mdelay, path)
            current_obj = obj(var, mdelay, mpath, hpath, head, flag=False)
            flag = True
        temp_t = t
        d, t = 0, 0
        for i in range(1,N):
            # Remove CH
            if H[i]:
                H[i] = False
                mpath, hpath, head = graph(var, delay, mdelay, path)
                temp_obj = obj(var, mdelay, mpath, hpath, head)
                if current_obj - temp_obj>=d:
                    d = current_obj - temp_obj
                    t = i
                    # print(' {}: {:.2f} = {}-{}'.format(i, current_obj-temp_obj, current_obj, temp_obj))
                H[i] = True
        # print('d:', d)
        if t>0: 
            H[t] = False
            flag = True
        if max(temp_t,t) <= 0: break
        # print(temp_t, t)
        # print('-> obj:', obj(var, mdelay, head), '\n')
    mpath, hpath, head = graph(var, delay, mdelay, path)
    # print('='*20)
    # print('\nFinal Result:')
    # obj(var, mdelay, head, flag=True)
    # print('CHs:', np.array(list(range(N)))[H])

    return sinr, per, delay, mdelay, mpath, hpath, head, flag

def constraint(x, lb, ub):
    if not lb is None:
        x = np.vectorize(max)(x, lb)
    if not ub is None:
        x = np.vectorize(min)(x, ub)
    return x

def descent(var, variable='position', idx=None):
    parameter = sim.parameter
    lb, ub = parameter['Pmin'], parameter['Pmax']
    b = 1.0
    Q1, P1 = var['position'], var['power']
    obj1, grad = objfromvar(var, grad=True)
    grad_Q, grad_P, _ = grad
    if idx is not None:
        P1 = P1[idx]
        grad_P = grad_P[idx]
    if variable == 'position' or variable =='both':
        Q2 = (Q1-b*grad_Q)
        var['position'] = Q2
    if variable == 'power' or variable =='both':
        P2 = constraint(P1-b*grad_P, lb, ub)
        if idx is None:
            var['power'] = P2
        else:
            var['power'][idx] = P2

    obj2, _ = objfromvar(var)

    while obj2<obj1 and b<=20480:
        obj1 = obj2
        b *= 2
        if variable == 'position' or variable =='both':
            Q2 = (Q1-b*grad_Q)
            var['position'] = Q2
        if variable == 'power' or variable =='both':
            P2 = constraint(P1-b*grad_P, lb, ub)
            if idx is None:
                var['power'] = P2
            else:
                var['power'][idx] = P2
        obj2, _ = objfromvar(var)
    if b==1: a = 0
    else: a = b/2
    obja, objb = obj1, obj2
    # print(obja, objb)
    while b-a>1e-5:  
        c = (a+b)/2
        if variable == 'position' or variable =='both':
            Q2 = (Q1-c*grad_Q)
            var['position'] = Q2
        if variable == 'power' or variable =='both':
            P2 = constraint(P1-c*grad_P, lb, ub)
            if idx is None:
                var['power'] = P2
            else:
                var['power'][idx] = P2
        objc, _= objfromvar(var)
        # print('objc:',objc, c)
        if objb > obja:
            b = c
            objb = objc
        else:
            a = c
            obja = objc
            # print('fuck',a)
    # print('alpha:', a)
    # print(grad1)
    # exit()
    if a==0: 
        if obja>objb: a = b
        # print('obj a - b:', obja-objb)
        # print('grad:', norm(grad_Q), norm(grad_P))
    if variable == 'position' or variable =='both':
        Q2 = (Q1-a*grad_Q)
        var['position'] = Q2
    if variable == 'power' or variable =='both':
        P2 = constraint(P1-a*grad_P, lb, ub)
        if idx is None:
            var['power'] = P2
        else:
            var['power'][idx] = P2
    ng = 0
    if variable == 'position' or variable =='both': ng += norm(grad_Q)
    if variable == 'power' or variable =='both': ng += norm(grad_P)
    return a, ng
    


def B_position_opt(var, it):
    flag = False
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    N = len(Q)

    obj_pre = 0
    # sinr, per, delay, mdelay, path = delay_graph(var)
    # mpath, hpath, head = graph(var, delay, mdelay, path)
    valname = ['J', 'F', 'W', 'Pt', 'Pm', 'd1', 'd2', 'd3']
    for i in range(20):
        print(' ', time.ctime())
        print('\tB{} iter: {}'.format(it, i))
        # obj_current = obj(var, mdelay, head, flag=False)
        # grad_Q, grad_P, grad_V = obj_grad(var, sinr, per, delay, mdelay, mpath, hpath, head)
        alpha, ng = descent(var, variable='position')
        # print('alpha:', alpha)

        sinr, per, delay, mdelay, path = delay_graph(var)
        mpath, hpath, head = graph(var, delay, mdelay, path)
        vals = obj(var, mdelay, mpath, hpath, head, flag=True)
        obj_current = vals[0]

        print('\t\tobj:', obj_current)
        print('\t\talpha:', alpha)
        print('\t\tgrad:', ng)
        for k,v in enumerate(valname):
            print('\t\t  {} = {}'.format(v, vals[k]))
        if i>2 and (np.abs(obj_current- obj_pre) < 1e-6) and ng<1e-4 or alpha<1e-5:
            print('\t  ###converged.')
            print('\t    alpha:', alpha)
            print('\t    grad:', ng)
            print('\t    dobj:', abs(obj_current-obj_pre))
            if i>1: 
                flag = True
            break
        obj_pre = obj_current
        if i%1==0:
            # print('path:', path)
            fname = sim.figpath+'/#{}-{}-B{}'.format(N, it, i+1)
            title = 'iter:{}-B{} obj: {:.3f} EE: {:.3f}, alpha:{:.5f}'.format(it, i+1, obj_current, vals[1], alpha)
            plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)
    return sinr, per, delay, mdelay, mpath, hpath, head, flag 

def C_power_opt(var, it):
    flag = False
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    N = len(Q)

    obj_pre = 0
    # sinr, per, delay, mdelay, path = delay_graph(var)
    # mpath, hpath, head = graph(var, delay, mdelay, path)
    valname = ['J', 'F', 'W', 'Pt', 'Pm', 'd1', 'd2', 'd3']
    for i in range(4*N):
        print(' ', time.ctime())
        print('\tC{} iter: {}'.format(it, i))
        # obj_current = obj(var, mdelay, head, flag=False)
        # grad_Q, grad_P, grad_V = obj_grad(var, sinr, per, delay, mdelay, mpath, hpath, head)
        alpha, ng = descent(var, variable='power', idx=None)
        # print('alpha:', alpha)

        sinr, per, delay, mdelay, path = delay_graph(var)
        mpath, hpath, head = graph(var, delay, mdelay, path)
        vals = obj(var, mdelay, mpath, hpath, head, flag=True)
        obj_current = vals[0]

        print('\t\tobj:', obj_current)
        print('\t\talpha:', alpha)
        print('\t\tgrad:', ng)
        for k,v in enumerate(valname):
            print('\t\t  {} = {}'.format(v, vals[k]))
        if i>2 and (np.abs(obj_current- obj_pre) < 1e-6) and ng<1e-4 or alpha<1e-4:
            print('\t  ###converged.')
            print('\t    alpha:', alpha)
            print('\t    grad:', ng)
            print('\t    dobj:', abs(obj_current-obj_pre))
            if i>1: flag = True
            break
        obj_pre = obj_current
        if it==0 and i%1==0:
            # print('path:', path)
            fname = sim.figpath+'/#{}-{}-C{}'.format(N, it, i+1)
            title = 'iter:{}-C{} obj: {:.3f} EE: {:.3f}, alpha:{:.5f}'.format(it, i+1, obj_current, vals[1], alpha)
            plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)
    return sinr, per, delay, mdelay, mpath, hpath, head, flag

def BC_position_power_opt(var, it):
    flag = False
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    N = len(Q)

    obj_pre = 0
    # sinr, per, delay, mdelay, path = delay_graph(var)
    # mpath, hpath, head = graph(var, delay, mdelay, path)
    valname = ['J', 'F', 'W', 'Pt', 'Pm', 'd1', 'd2', 'd3']
    for i in range(20):
        print(' ', time.ctime())
        print('\tBC{} iter: {}'.format(it, i))
        # obj_current = obj(var, mdelay, head, flag=False)
        # grad_Q, grad_P, grad_V = obj_grad(var, sinr, per, delay, mdelay, mpath, hpath, head)
        alpha, ng = descent(var, variable='both')
        # print('alpha:', alpha)

        sinr, per, delay, mdelay, path = delay_graph(var)
        mpath, hpath, head = graph(var, delay, mdelay, path)
        vals = obj(var, mdelay, mpath, hpath, head, flag=True)
        obj_current = vals[0]

        print('\t\tobj:', obj_current)
        print('\t\talpha:', alpha)
        print('\t\tgrad:', ng)
        for k,v in enumerate(valname):
            print('\t\t  {} = {}'.format(v, vals[k]))
        if i>2 and (np.abs(obj_current- obj_pre) < 1e-6) and ng<1e-4 or alpha<1e-5:
            print('\t  ###converged.')
            print('\t    alpha:', alpha)
            print('\t    grad:', ng)
            print('\t    dobj:', abs(obj_current-obj_pre))
            if i>1: flag = True
            break
        obj_pre = obj_current
        if i%1==0:
            # print('path:', path)
            fname = sim.figpath+'/#{}-{}-B{}'.format(N, it, i+1)
            title = 'iter:{}-B{} obj: {:.3f} EE: {:.3f}, alpha:{:.5f}'.format(it, i+1, obj_current, vals[1], alpha)
            plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)
    return sinr, per, delay, mdelay, mpath, hpath, head, flag

def block_coordinate_descent(var):
    '''
    input:
        number of nodes: N
    output:
        Q = [q1,q2,...,qN] Nx2
        P = [p1,p2,...,pN] 1xN
        H = [1,0,....,0,1] 1xN
        V = [v1, v2] 1x2
    '''
    # keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']
    print('start:', time.ctime())
    #var = initialize()
    N = len(var['power'])
    max_iter = 100
    obj_values = []
    allvals = []
    flagA, flagB, flagC, flagBC = False, False, False, False
    for t in range(max_iter):
        print('Iteration:', t)

        print('  A: Clustering...')
        sinr, per, delay, mdelay, mpath, hpath, head, flagA = A_clustering(var)
        # fname = 'img/#{}-{}-A-{}'.format(N, t, 0)
        # title = 'iter:A{}-{} obj: {:.5f}'.format(t, 0, obj(var, mdelay, mpath, hpath, head, flag=False))
        # plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)
        if t==0:
            vals = obj(var, mdelay, mpath, hpath, head, flag=True)
            obj_values.append(vals[0])
            allvals.append(vals)
            fname = sim.figpath+'/#{}-{}'.format(N, t)
            title = 'iter:{} obj: {:.5f}'.format(t, vals[0])
            plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)

        print('  B: Position Optimizing')
        # sinr, per, delay, mdelay, mpath, hpath, head, flagB = B_position_opt(var, t)

        print('  C: Power Optimizing')
        # sinr, per, delay, mdelay, mpath, hpath, head, flagC = C_power_opt(var, t)
        # print(delay)

        print('  BC: Position & Power Optimizing')
        sinr, per, delay, mdelay, mpath, hpath, head, flagBC = BC_position_power_opt(var, t)

        vals = obj(var, mdelay, mpath, hpath, head, flag=True)
        obj_values.append(vals[0])
        allvals.append(vals)
        d = vals[5:8]
        print('='*40)
        fname = sim.figpath+'/#{}-{}'.format(N, t+1)
        title = 'iter:{} obj: {:.5f}'.format(t+1, vals[0])
        # print(title, fname)
        plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)
        print('now:', time.ctime())
        if t>3 and abs(obj_values[-1]-obj_values[-4])<1e-6 and np.all(d<1e-4) or (~flagA and ~flagB and ~flagC and ~flagBC):
            print('!!!converged in iteration', t)
            break
    return sinr, per, delay, mdelay, mpath, hpath, head, obj_values, allvals

def sec2date(sec):
    sec = int(sec)
    return '{}:{}:{}'.format(sec//3600, sec//60%60, sec%60)

def greedy_descent(var, msg, timestamp):
    debug = False
    # print('start:', time.ctime())
    N = len(var['power'])
    max_iter = 100
    allvals = []
    sinr, per, delay, mdelay, path = delay_graph(var)
    mpath, hpath, head = graph(var, delay, mdelay, path)
    # keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']
    vals = obj(var, mdelay, mpath, hpath, head, flag=True)
    allvals.append(vals)
    valname = ['J', 'F', 'W', 'Pt', 'Pm', 'd1', 'd2', 'd3']
    flagA = False
    for t in range(max_iter):
        # cost_time = sim.difftime()
        cost_time = sec2date(time.time()-timestamp)
        if t%10==0:
            print(msg, 'Iteration: {:02d}'.format(t), '\t', cost_time)

        if debug:
            print('  Step A: Clustering...')
        if t%10 == 0:
            sinr, per, delay, mdelay, mpath, hpath, head, flagA = A_clustering(var)

        if debug:
            print('  Step B&C: Position & Power Optimizing...')

        vb = 'both' if sim.parameter['P']<=1 else 'position'
        alpha, ng = descent(var, variable=vb)
        # print('alpha:', alpha)

        sinr, per, delay, mdelay, path = delay_graph(var)
        mpath, hpath, head = graph(var, delay, mdelay, path)
        # keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']
        vals = obj(var, mdelay, mpath, hpath, head, flag=True)
        allvals.append(vals)
        obj_current = vals[0]

        if debug:
            print('\tobj:', obj_current)
            print('\talpha:', alpha)
            print('\tgrad:', ng)
            for k,v in enumerate(valname):
                print('\t\t  {} = {}'.format(v, vals[k]))

        dobj = abs(allvals[-1][0]-allvals[-2][0])

        if ((dobj < 1e-6) and ng<1e-4 or alpha<1e-5) and ~flagA:
            print(msg, '\t###converged.')
            print(msg, '\talpha:', alpha)
            print(msg, '\tgrad:', ng)
            print(msg, '\tdobj:', dobj)
            break
        # if t<10 or t%5==0:
        if 0:
            # print('path:', path)
            fname = sim.figpath+'/#{}-{}'.format(N, t)
            title = 'iter:{} obj: {:.3f} EE: {:.3f}, alpha:{:.5f}'.format(t, obj_current, vals[1], alpha)
            plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, filename=fname, title=title)

    return sinr, per, delay, mdelay, mpath, hpath, head, allvals