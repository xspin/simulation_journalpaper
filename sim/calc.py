#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from sim.graph import graph, SINR_PER_matrix, gain, delay_graph, per_an, per_gn, plot
import sim

def delay_grad_matrix(var, sinr, per, delay):
    # 1-hop delay
    parameter = sim.parameter
    Q, P = var['position'], var['power']
    T = parameter['RTT']
    N = len(P)
    grad_Q = np.zeros([N,N,N,2]) #i,j,k
    grad_P = np.zeros([N,N,N])
    for i in range(N):
        for j in range(N):
            if i==j or delay[i,j]==np.inf: continue 
            for k in range(N):
                pQ, pP = per_grad(var, sinr[i,j],i,j,k)
                grad_Q[i,j,k] = T*pQ/(1-per[i,j])**2
                grad_P[i,j,k] = T*pP/(1-per[i,j])**2
    return grad_Q, grad_P

def delay_grad_check(var):
    print('='*30)
    print('Grad Check: Delay')
    Q, P = var['position'], var['power']
    sinr, per, delay, mdelay, path = delay_graph(var)
    # mpath, hpath, head = graph(var, delay, mdelay, path)
    grad_Q, grad_P = delay_grad_matrix(var, sinr, per, delay)
    N = len(path)
    epsilon = 0.0000001
    dQ = 0
    # gradient on Q
    for k in range(1,N):
        for t in range(2):
            Q[k, t] += epsilon
            sinr1, per1, delay1, mdelay1, path1 = delay_graph(var)
            Q[k, t] -= 2*epsilon
            sinr2, per2, delay2, mdelay2, path2 = delay_graph(var)
            Q[k, t] += epsilon
            for i in range(N):
                for j in range(N):
                    if i==j or delay[i,j]==np.inf: continue 
                    temp = (delay1[i,j]-delay2[i,j])/(2*epsilon)
                    dQ += (temp-grad_Q[i,j,k][t])**2
    dQ = np.sqrt(dQ)/np.prod(grad_Q.shape)
    print('  dQ:{}, ng:{:.6f}'.format(dQ, norm(grad_Q)))
    # gradient on P
    dP = 0
    for k in range(N):
        P[k] += epsilon
        sinr1, per1, delay1, mdelay1, path1 = delay_graph(var)
        P[k] -= 2*epsilon
        sinr2, per2, delay2, mdelay2, path2 = delay_graph(var)
        P[k] += epsilon
        for i in range(N):
            for j in range(N):
                if i==j or delay[i,j]==np.inf: continue 
                temp = (delay1[i,j]-delay2[i,j])/(2*epsilon)
                dP += (temp-grad_P[i,j,k])**2
    dP = np.sqrt(dP)/np.prod(grad_P.shape)
    print('  dP:{}, ng:{:.6f}'.format(dP, norm(grad_P)))
    if dQ<1e-6 and dP<1e-6: print('Pass.')
    else: print('Not Pass.')

def gain_grad(var, i, j, k):
    parameter = sim.parameter
    Q = var['position']
    rho = parameter['rho']
    d = np.sum(np.square(Q[i]-Q[j]))**2
    t = 2*rho*(Q[i]-Q[j])/d
    if k==j: return t
    elif k==i: return -t 
    else: return np.array([0,0]) 

def gain_grad_check(var):
    print('='*30)
    print('Grad Check: gain')
    Q = var['position']
    N = len(Q)
    eps = 0.01 
    delta = 0
    for i in range(N):
        for j in range(N):
            if i==j: continue
            for k in range(N):
                grad1 = gain_grad(var, i, j, k)
                for t in range(2):
                    Q[k,t] += eps
                    g1 = gain(var,i,j)
                    Q[k,t] -= 2*eps
                    g2 = gain(var,i,j)
                    Q[k,t] += eps
                    grad2 = (g1-g2)/(2*eps)
                    delta += (grad1[t]-grad2)**2
    dQ = np.sqrt(delta)/N
    print('  dQ:', np.sqrt(delta)/N)
    if dQ<1e-6:
        print('Pass.')
    else: print('Not Pass.')

def sinr_grad(var, i, j, k):
    parameter = sim.parameter
    Q, P = var['position'], var['power']
    prob = parameter['probability']
    noise = parameter['noise']
    d1 = gain_grad(var,i,j,k)/gain(var,i,j)
    temp1 = 0
    temp2 = 0
    for t in range(len(P)):
        if t==i or t==j: continue
        temp1 += prob*P[t]*gain_grad(var,t,j,k)
        temp2 += prob*P[t]*gain(var,t,j)
    temp = d1 - temp1/(temp2+noise)
    pq = temp*10/np.log(10)
    if k==0: pq = np.array([0,0])
    if k==j: pp = 0
    elif k==i: pp = 10/np.log(10)/P[i]
    else: pp = -10*prob*gain(var,k,j)/np.log(10)/(temp2+noise)
    return pq, pp

def sinr_per_grad_check(var):
    print('='*30)
    print('Grad Check: SINR & PER')
    Q = var['position']
    P = var['power']
    N = len(Q)
    eps = 0.000001 
    dQ1, dQ2 = 0, 0
    sinr, per = SINR_PER_matrix(var)
    # gradient on Q
    for k in range(1, N):
        for t in range(2):
            Q[k,t] += eps
            # q1 = np.copy(Q)
            sinr1, per1 = SINR_PER_matrix(var)
            Q[k,t] -= 2*eps
            # q2 = np.copy(Q)
            sinr2, per2 = SINR_PER_matrix(var)
            # print(norm((sinr1-sinr2)/(2*eps)), norm(q1-q2))
            Q[k,t] += eps
            for i in range(N):
                for j in range(N):
                    if i==j: continue
                    grad,_ = sinr_grad(var, i, j, k)
                    gradper,_ = per_grad(var, sinr[i,j], i,j,k)
                    grad1 = (sinr1[i,j]-sinr2[i,j])/(2*eps)
                    grad2 = (per1[i,j]-per2[i,j])/(2*eps)
                    d1 = (grad[t]-grad1)**2
                    d2 = (gradper[t]-grad2)**2
                    dQ1 += d1
                    dQ2 += d2
                    # print('{}, {}, {}: {:.5f} {:.5f} {:.8f}'.format(i,j,k,grad[t],grad1,d))
        if k==0: 
            print('  pQ0:  dQ_sinr:', np.sqrt(dQ1), 'dQ_per:', np.sqrt(dQ2))
    print('  dQ_sinr:', np.sqrt(dQ1)/N, ', dQ_per:', np.sqrt(dQ2)/N)
    dP1, dP2 = 0, 0
    # gradient on P
    for k in range(N):
        P[k] += eps
        sinr1, per1 = SINR_PER_matrix(var)
        P[k] -= 2*eps
        sinr2, per2 = SINR_PER_matrix(var)
        P[k] += eps
        for i in range(N):
            for j in range(N):
                if i==j: continue
                _ ,grad = sinr_grad(var, i, j, k)
                _, gradper = per_grad(var, sinr[i,j], i,j,k)
                grad1 = (sinr1[i,j]-sinr2[i,j])/(2*eps)
                grad2 = (per1[i,j]-per2[i,j])/(2*eps)
                d1 = (grad-grad1)**2
                d2 = (gradper-grad2)**2
                dP1 += d1
                dP2 += d2
                # print('{}, {}, {}: {:.5f} {:.5f} {:.5f}'.format(i,j,k,grad,grad1,d))
    print('  dP_sinr:', np.sqrt(dP1)/N, ', dP_per:', np.sqrt(dP2)/N)

def per_grad(var, sinrij, i, j, k):
    an, gn = per_an, per_gn
    pQ, pP = sinr_grad(var,i,j,k)
    t = 0
    if sinrij > np.log(an)/gn:
        t = -gn*an*np.exp(-gn*sinrij)
    return t*pQ, t*pP

def penalty(var, mdelay, mpath, hpath, head):
    parameter = sim.parameter
    H, Q = var['head'], var['position']
    N = len(H)
    sm = parameter['delay_h2m']
    sh = parameter['delay_l2h']
    lmd1 = parameter['lambda1']
    lmd2 = parameter['lambda2']
    lmd3 = parameter['lambda3']
    delta1 = 0
    delta2 = 0
    delta3 = 0
    n1, n2, n3, nh1 = 0, 0, 0, 0
    for i in range(1, N):
        if H[i]:
            if hpath[i]<0:
                delta3 += norm(Q[i]-Q[0])
                n3 += 1
                nh1 += 1
            else:
                # delta2 += 1-1/(1+max(0, mdelay[0,i] - sh))
                delta2 += max(0, mdelay[0,i] - sh)
                if mdelay[0,i]>sh: n2 += 1
        else:
            j = head[i]
            if j>=0:
                # delta1 += 1-1/(1+max(0, mdelay[j,i] - sm))
                delta1 += max(0, mdelay[j,i] - sm)
                if mdelay[j,i]>sm: n1 += 1
            else:
                delta3 += norm(Q[i]-Q[0])
                n3 += 1
    return lmd1*delta1, lmd2*delta2, lmd3*delta3, n1, n2, n3, nh1

def penalty_grad(var, sinr, per, delay, mdelay, mpath, hpath, head):
    parameter = sim.parameter
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    sm = parameter['delay_h2m']
    sh = parameter['delay_l2h']
    lmd1 = parameter['lambda1']
    lmd2 = parameter['lambda2']
    lmd3 = parameter['lambda3']
    N = len(H)
    delay_pQ, delay_pP = delay_grad_matrix(var, sinr, per, delay)
    # derivation of pernalty
    pQ1 = np.zeros([N,2])
    pQ2 = np.zeros([N,2])
    pQ3 = np.zeros([N,2])
    pP1, pP2 = np.zeros(N), np.zeros(N)
    for k in range(N):
        for i in range(N):
            if H[i]:
                if i==0 or mdelay[0,i]<=sh: continue
                j = i
                while hpath[j]>=0: 
                    t = j
                    j = hpath[j]
                    pQ2[k] += delay_pQ[j,t,k]*lmd2 
                    pP2[k] += delay_pP[j,t,k]*lmd2
            else:
                j = head[i]
                if j<0 or mdelay[j,i]<=sm: continue
                j = i
                while mpath[j]>=0: 
                    t = j
                    j = mpath[j]
                    pQ1[k] += delay_pQ[j,t,k]*lmd1 
                    pP1[k] += delay_pP[j,t,k]*lmd1
    for i in range(1,N):
        if H[i]:
            if hpath[i]<0:
                pQ3[i] += (Q[i]-Q[0])/norm(Q[i]-Q[0])*lmd3
        elif head[i]<0:
                pQ3[i] += (Q[i]-Q[0])/norm(Q[i]-Q[0])*lmd3
    return pQ1, pQ2, pQ3, pP1, pP2

def penalty_grad_check(var):
    print('='*30)
    print('Grad Check: Penalty')
    Q, P = var['position'], var['power']
    N = len(Q)
    sinr, per, delay, mdelay, path = delay_graph(var)
    mpath, hpath, head = graph(var, delay, mdelay, path)

    epsilon = 0.0001
    
    delta = penalty(var, mdelay, mpath, hpath, head)
    pd = penalty_grad(var, sinr, per, delay, mdelay, mpath, hpath, head)
    dQ = [0,0,0]
    for k in range(1, N):
        for i in range(2):
            Q[k, i] += epsilon
            sinr, per, delay, mdelay, path = delay_graph(var)
            mpath, hpath, head = graph(var, delay, mdelay, path)
            da = penalty(var, mdelay, mpath, hpath, head)
            Q[k, i] -= 2*epsilon
            sinr, per, delay, mdelay, path = delay_graph(var)
            mpath, hpath, head = graph(var, delay, mdelay, path)
            db = penalty(var, mdelay, mpath, hpath, head)
            Q[k, i] += epsilon
            for t in range(3):
                temp = (da[t]-db[t])/(2*epsilon)
                dQ[t] += (temp-pd[t][k,i])**2
            # print('{},{}:'.format(k,i), pQ[k,i], pw)
    for t in range(3):
        print('  Delta', t+1)
        dQ[t] = np.sqrt(dQ[t])/np.prod(pd[t].shape)
        print('\tdQ:{:.6f}, ng:{:.6f}'.format(dQ[t], norm(pd[t])))
    print('-'*10)
    # gradient on P
    gP = pd[3]
    dP = [0,0]
    for k in range(N):
        P[k] += epsilon
        sinr, per, delay, mdelay, path = delay_graph(var)
        mpath, hpath, head = graph(var, delay, mdelay, path)
        da = penalty(var, mdelay, mpath, hpath, head)
        P[k] -= 2*epsilon
        sinr, per, delay, mdelay, path = delay_graph(var)
        mpath, hpath, head = graph(var, delay, mdelay, path)
        db = penalty(var, mdelay, mpath, hpath, head)
        P[k] += epsilon
        for t in range(2):
            temp = (da[t]-db[t])/(2*epsilon)
            dP[t] += (temp-pd[t+3][k])**2
        # print('{},{}:'.format(k,i), pQ[k,i], pw)
    for t in range(2):
        print('  Delta', t+1)
        dP[t] = np.sqrt(dP[t])/np.prod(pd[3+t].shape)
        print('\tdP:{:.6f}, ng:{:.6f}'.format(dP[t], norm(pd[t+3])))
    print('-'*30)

def structure_width(var):
    parameter = sim.parameter
    Q, V = var['position'], var['velocity']
    R = parameter['RC']
    v_vert = np.r_[V[1], -V[0]]/norm(V)
    u = np.dot(Q, v_vert)
    u = np.sort(u)
    du = u[1:] - u[:-1]
    w = 2*R + np.sum(np.vectorize(min)(du, 2*R))
    # w1 = 2*R+len(du)*R +(u[-1]-u[0])/2 - np.sum(np.abs(du/2-R))
    return w

def structure_width_grad(var):
    parameter = sim.parameter
    Q, V = var['position'], var['velocity']
    R = parameter['RC']
    v_vert = np.r_[V[1], -V[0]]/norm(V)
    # print('v_vert',v_vert)
    u = np.dot(Q, v_vert)
    idx = np.argsort(u)
    idx1 = np.zeros(len(idx), dtype=int)
    for i, j in enumerate(idx): idx1[j] = i
    u = u[idx]
    du = u[1:] - u[:-1]
    # print('u:', u, 'du:', du, '2R:',2*R)
    f = lambda x: 0.5 if x>=2*R else -0.5
    du2 = np.r_[3*R, du, 3*R] 
    du2 = np.vectorize(f)(du2)
    pwpu = du2[1:]-du2[:-1]
    pwpu = pwpu[idx1]
    # print('pwpu:', pwpu)
    # pwpQ = np.array([x*v_vert for x in pwpu])
    pwpQ = np.outer(pwpu, v_vert)
    # pwpv = np.dot(pwpu, Q.dot([[0,1],[-1,0]]))/norm(V)
    v = norm(V) 
    # pwpv = np.dot(pwpu, Q.dot([[0,1],[-1,0]]).dot(1/v-V/(v**3)))
    I = np.eye(2)
    pwpv = (1/v*I-V[:,None]*V/(v**3)).dot([[0,-1],[1,0]]).dot(Q.T).dot(pwpu.T)
    # print('pwpv', pwpv)
    return pwpQ, pwpv

def width_grad_check(var):
    print('='*30)
    print('Grad Check: width')
    Q, V = var['position'], var['velocity']
    epsilon = 0.00001
    pQ, pv = structure_width_grad(var)
    dv = 0
    for i in range(len(V)):
        break
        V[i] += epsilon
        w1 = structure_width(var)
        V[i] -= 2*epsilon
        w2 = structure_width(var)
        pw = (w1-w2)/(2*epsilon)
        dv += (pv[i]-pw)**2
        print(' {}:'.format(i), pv[i], pw)
        V[i] += epsilon
    dv /= len(V)
    dQ = 0
    for k in range(len(Q)):
        for i in range(2):
            Q[k, i] += epsilon
            w1 = structure_width(var)
            Q[k, i] -= 2*epsilon
            w2 = structure_width(var)
            pw = (w1-w2)/(2*epsilon)
            dQ += (pQ[k,i]-pw)**2
            # print('{},{}:'.format(k,i), pQ[k,i], pw)
            Q[k, i] += epsilon
    dQ /= np.prod(Q.shape)
    print('  Q:{}'.format(dQ))
    print('-'*30)
def structure_area(var):
    parameter = sim.parameter
    Q, V = var['position'], var['velocity']
    R = parameter['RC']
    N = len(Q)
    ds = 0
    for i in range(N):
        for j in range(i+1, N):
            d = norm(Q[i]-Q[j])
            if d < 2*R:
                ds += 2*np.arccos(d/(2*R))*R*R-d*np.sqrt(R*R-d*d/4)
    return N*np.pi*R*R-ds

def overlap_grad(var, i, j):
    parameter = sim.parameter
    Q = var['position']
    R = parameter['RC']
    if i==j: return 0
    d = norm(Q[i]-Q[j])
    if d>=2*R: return 0
    return -np.sqrt(4*R*R-d*d)/d*(Q[i]-Q[j]) 

def structure_area_grad(var):
    parameter = sim.parameter
    Q, V = var['position'], var['velocity']
    R = parameter['RC']
    N = len(Q)
    pSpQ = np.zeros([N,2])
    for i in range(N):
        for j in range(N):
            if j==i: continue
            # d = norm(Q[i]-Q[j])
            pSpQ[i] += overlap_grad(var, i, j) 
        pSpQ[i] = -pSpQ[i]
    return pSpQ, [0,0]

def area_grad_check(var):
    print('='*30)
    print('Grad Check: Area')
    Q, V = var['position'], var['velocity']
    epsilon = 0.00001
    pQ, pv = structure_area_grad(var)
    dv = 0
    for i in range(len(V)):
        break
        V[i] += epsilon
        w1 = structure_area(var)
        V[i] -= 2*epsilon
        w2 = structure_area(var)
        pw = (w1-w2)/(2*epsilon)
        dv += (pv[i]-pw)**2
        print(' {}:'.format(i), pv[i], pw)
        V[i] += epsilon
    dv /= len(V)
    dQ = 0
    for k in range(len(Q)):
        for i in range(2):
            Q[k, i] += epsilon
            w1 = structure_area(var)
            Q[k, i] -= 2*epsilon
            w2 = structure_area(var)
            pw = (w1-w2)/(2*epsilon)
            dQ += (pQ[k,i]-pw)**2
            # print('{},{}:'.format(k,i), pQ[k,i], pw)
            Q[k, i] += epsilon
    dQ /= np.prod(Q.shape)
    print('  Q:{}'.format(dQ))
    print('-'*30)


def motion_power(var):
    parameter = sim.parameter
    V = var['velocity']
    v = norm(V)
    cf = parameter['c_force']
    cr = parameter['c_resis']
    m = parameter['m']
    g = parameter['g']
    N = parameter['#nodes']
    Pm = cf * np.sqrt(cr**2*v**4+(m*g)**2)
    return N*Pm
    
def motion_power_grad(var):
    parameter = sim.parameter
    V = var['velocity']
    v2 = np.sum(np.square(V))
    cf = parameter['c_force']
    cr = parameter['c_resis']
    m = parameter['m']
    g = parameter['g']
    N = parameter['#nodes']
    return 2*N*cf*cr*cr*v2/np.sqrt(cr*cr*v2*v2+(m*g)**2)*V

def obj(var, mdelay, mpath, hpath, head, flag=False):
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    a = sim.parameter['a']
    R = sim.parameter['RC']
    Pmax = sim.parameter['Pmax']
    R = sim.parameter['RC']
    N = len(Q)
    S = structure_area(var) 
    W = structure_width(var)
    # Cov = a*S + (1-a)*W
    # Pm = motion_power(var)
    Fs = S/(N*R*R*np.pi) 
    Fw = W/(N*R*2)
    F = a*Fs + (1-a)*Fw
    Pt = np.sum(P)
    p = Pt/(N)
    Pm = 0
    d1, d2, d3, n1, n2, n3, nh1 = penalty(var, mdelay, mpath, hpath, head)
    J = 1.0/F+d1+d2+d3
    vals = J
    if flag:
        # print('  Coverage Width (m):', w)
        # print('  Penalty-H2M:', d1, ', Penalty-L2H:', d2)
        # print('  Energy Efficiency (m^2/J):', F)
        # print('  Obj Value:', J)
        vals = np.r_[J, F, Fs, Fw, p, d1, d2, d3, n1, n2, n3, nh1, sum(H)]
    return vals 
    # keys = ['J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']

def objfromvar(var, grad=False):
    sinr, per, delay, mdelay, path = delay_graph(var)
    mpath, hpath, head = graph(var, delay, mdelay, path)
    obj_current = obj(var, mdelay, mpath, hpath, head, flag=False)
    if grad:
        grad = obj_grad(var, sinr, per, delay, mdelay, mpath, hpath, head)
    return obj_current, grad

def obj_grad(var, sinr, per, delay, mdelay, mpath, hpath, head):
    parameter = sim.parameter
    Q, P, H, V = var['position'], var['power'], var['head'], var['velocity']
    # sm = parameter['delay_h2m']
    # sh = parameter['delay_l2h']
    # lmd1 = parameter['lambda1']
    # lmd2 = parameter['lambda2']
    # lmd3 = parameter['lambda3']
    a = parameter['a']
    R = parameter['RC']
    # Pmax = parameter['Pmax']
    N = len(P)
    S = structure_area(var) 
    W = structure_width(var)
    Fs = S/(N*R*R*np.pi) 
    Fw = W/(N*R*2)
    F = a*Fs + (1-a)*Fw
    # Pt = np.sum(P)
    # Pm = motion_power(var)
    width_pQ1, width_pV1 = structure_area_grad(var)
    width_pQ2, width_pV2 = structure_width_grad(var)
    F_pQ = a*width_pQ1/(N*np.pi*R*R) + (1-a)*width_pQ2/(N*R*2)

    pQ1, pQ2, pQ3, pP1, pP2 = penalty_grad(var, sinr, per, delay, mdelay, mpath, hpath, head)
    grad_Q = np.zeros([N,2])
    grad_P = np.zeros(N)
    for k in range(N):
        # gradient on position Q
        grad_Q[k] = -F_pQ[k]/(F*F)+pQ1[k]+pQ2[k]+pQ3[k]
        # gradient on power P
        # grad_P[k] = 1.0/(N*Pmax) + pP1[k]+pP2[k]
        grad_P[k] = pP1[k]+pP2[k]
    # gradient on velocity V
    # power_pV = motion_power_grad(var)
    # grad_V = power_pV/(v*w) + (Pt+Pm)/(v*w)**2*(v*power_pV+w*V/v)
    grad_V = [0,0]
    grad_Q[0] = [0,0]
    # print(grad_Q)
    # print(width_pQ)
    # print(width_pQ/ Pt)
    # exit()
    return grad_Q, grad_P, grad_V

def obj_grad_check(var):
    print('='*30)
    print('Grad Check: obj')
    Q, P = var['position'], var['power']
    N = len(Q)
    sinr, per, delay, mdelay, path = delay_graph(var)
    mpath, hpath, head = graph(var, delay, mdelay, path)
    # plot(var, sinr, per, delay, mdelay, mpath, hpath, head, show=False, save=True, title='Initial State', filename='img/!%d-gradcheck.png'%(N))

    # print(var['head'])
    epsilon = 0.0001
    _, pg = objfromvar(var, grad=True)
    pQ = pg[0]
    dQ = 0
    for k in range(1, len(Q)):
        for i in range(2):
            Q[k, i] += epsilon
            w1, _ = objfromvar(var)
            Q[k, i] -= 2*epsilon
            w2, _ = objfromvar(var)
            pw = (w1-w2)/(2*epsilon)
            # print(pQ[k, i],',' , pw, end=' | ')
            dQ += (pQ[k,i]-pw)**2
            # print('{},{}:'.format(k,i), pQ[k,i], pw)
            Q[k, i] += epsilon
        # print('')
    dQ = np.sqrt(dQ)/np.prod(Q.shape)
    print('  dQ:{}, npQ:{:.6f}'.format(dQ, norm(pQ)))

    # gradient on P
    dP = 0
    pP = pg[1]
    for k in range(1, len(Q)):
        P[k] += epsilon
        w1, _ = objfromvar(var)
        P[k] -= 2*epsilon
        w2, _ = objfromvar(var)
        pw = (w1-w2)/(2*epsilon)
        Q[k, i] += epsilon
        # print(pQ[k, i],',' , pw, end=' | ')
        dP += (pP[k]-pw)**2
        # print('{},{}:'.format(k,i), pQ[k,i], pw)
        # print('')
    dP = np.sqrt(dP)/np.prod(pP.shape)
    print('  dP:{}, npP:{:.6f}'.format(dP, norm(pP)))
    print('-'*30)
