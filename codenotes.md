#### 参数

- var
  - position, power, velocity, CH
- mdelay
  - mdelay[i,j] is the muti-hop delay from i to j
- mpath
  - mpath[i] = j if the previous hop is j from the corresponding CH
- hpath
  - hpath[i] = j if the previous hop is j from the leader
- head
  - head[i] = j if CM i's CH is j otherwise 0

#### 仿真数据

Number of Clusters vs #Nodes

- Cover Radius
- Static/Optimized Power
- Delay_h2m, Delay_l2h, Delay_h2m/Delay_l2h
- interference probability

Power Efficiency vs #Nodes

- Cover Radius
- Static/Optimized Power
- ​

Area+Width vs #Nodes

- Cover Radius
- Static/Optimized Power
- weight

Average area vs #Nodes



N, J, F, S, W, Pt, d1, d2, d3, #CH, #CM', #CH', #DN

keys = ['N', 'J', 'F', 'S', 'W', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3', 'NH1', 'NH']

- N1-N3 对应 D1-D3中的节点数
- NH 为CH数
- NH1为N3中的CH数（未连通）



Log File:

- YMD-H:M:S
- N, R, r, a, $\sigma_M$, $\sigma_H$, tp
- J, F, S, W, P, D1, D2, D3, N1, N2, N3, NH1, NH

figures/

- R
- St/Op Power
- a
- sigma
- ​



K: N1-N3, NH1, NH

S: S, Sav

P: P, Pav

N = 10, 20, 30,40, 50

Default parameters:

- R = 50

- r=0.05

- a=1.0

- sm, sh = 4, 6

- popt = T

- Pm = 100

  ​

#### Fig A: Radius (R)

- R = 20, 50, 80, 110
- F, S, W: the coverage efficiency degrades as R increases. CE decrease as N increase

#### Fig B: Delay Threshold (s)

- sm,sh = (2,3), (4,6), (6,9), (8, 12)
- F/S/W:  the coverage efficiency increases as $\sigma_{H/M}$ increase
- N123: the number of the nodes that is unsatisfiled the constraints increase as it decreases
- CH: decrease as s increase
- delays: avg, var
- powers: avg,var

#### Fig C: Transmit Power (p)

- Opt Power: True, 30,60,90
- F/S/W: without power optimization, the CE increases as power increases. With power optimization, the CE can achieve a high value with a lower average power
- Powers: avg, var
- CH: decrease as P increase
- N123: with the power optimization, there are less the nodes that ...

#### Fig D: Interference (r)

- interference r: 0.01, 0.02, 0.05, 0.1
- F: decreases as r increases
- CH: increases as r increases
- N123: increases as r increases
- delays: a_avg, a_var
- p_var

#### Fig E: Coverage Weight (a) [omit]

- a = 0, 0.3, 0.6, 1.0
- Fs: increase as a increase, Fw decrease (lines)
- ​





Ready to do

- delay statistics (a,h,m): mean, variance, min, max
- Pi: random with Gaussian distribution, mean, var, min, max
- a sample of the evolution of the state and the values
- multiply threads optimization
- related works
- algorithm + proof + derivation
- simulation

