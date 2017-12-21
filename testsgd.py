from SGD2 import Sgd
import numpy as np
import time
import sgdtestdatapre as sdp

Thrsrp = -88
Thsinr = -3
Thdis = 1
tPow = 18.2
'''
def readdata(path):
    f1=open(r"E:\2017.10.01.CMDI\2017.10.01.CMDI\%s.txt"%path)
    s=f1.readlines()
    a = []
    for line in s:
        line =line.rstrip()
        b = [np.float32(i) for i in line.split('\t')]
        a.append(b)
    return np.array(a)
t1 = time.time()
pathloss = readdata("pathloss")
dis = readdata("dis")
alpha = readdata("alpha")
beta = readdata("beta")
tpow = readdata("tpow")
t2 = time.time()
#print (dis.shape)
'''
pathloss, dis, alpha1, beta1, tpow, alpha2, beta2, covered_s_count = sdp.getdata()
print(covered_s_count)
#pathloss, dis, alpha1, beta1, tpow, alpha2, beta2 = sdp.getdata()
#t3 = time.time()
#print(t2 - t1, t3 - t2)

a = Sgd(dis, alpha1, beta1, Thrsrp, Thsinr, Thdis, tpow, pathloss, alpha2, beta2)
#print(pathloss.shape,dis.shape,alpha.shape,beta.shape,tpow.shape)
#print(covered_s_count)
#for i in [0.01,0.03,0.1,0.3,0.5,1]:
i=0.06
alpha, beta, coverarea = a.main(i)
print(i,"coverarea is %f" %coverarea)
