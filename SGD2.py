#!python2
#coding=utf-8
import numpy as np
from numpy import cos, sin
from math import pi
import random
class Sgd():
    def __init__(self, dis, alpha1, beta1, Thrsrp, Thsinr, Thdis, tPow, pathloss, alpha2, beta2):
        """
            dis：距离, alpha：方向角, beta：下倾角, pathloss：路损 为 采样点数*扇区数矩阵        
            tpow：是扇区发射功率(1 * 扇区数)
        """
        self.dis = dis
        self.alpha = alpha2
        self.beta = beta2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_samples = dis.shape[0]
        self.num_sectors = dis.shape[1]
        self.dalpha = np.zeros(alpha1.shape)
        self.dbeta = np.zeros(beta1.shape)
        self.Thrsrp = Thrsrp
        self.Thsinr = Thsinr
        self.Thdis = Thdis
        self.pathloss = pathloss
        self.tPow = tPow
        self.rTh = Thrsrp
        self.sTh = Thsinr
        self.eNum = 1
        self.fNum = 200
        self.noise = -110
        

        
        
    def getrsri(self, X, a): 
        alpha = self.alpha
        beta = self.beta
        dis = self.dis

        #print(alpha)
        darssi = np.zeros(self.num_samples * self.num_sectors).reshape(self.num_samples, -1)
        darssi[a,:] = self.galpha(alpha[a,:], beta[a,:]) * X[a,:]#计算方向角 
        darssi[dis > self.Thdis] = 0.0
        #darssi[a,:][dis[a,:] < self.Thdis] = self.galpha(alpha[a,:][dis[a,:] < self.Thdis], beta[a,:][dis[a,:] < self.Thdis]) * X[a,:][dis[a,:] < self.Thdis]#计算方向角 
        #print(darssi[a,:][dis[a,:] < self.Thdis])
        dbrssi = np.zeros(self.num_samples * self.num_sectors).reshape(self.num_samples, -1)
        dbrssi[a,:] = self.gbeta(alpha[a,:], beta[a,:]) * X[a,:]
        dbrssi[dis > self.Thdis] = 0.0
        #dbrssi[a,:][dis[a,:] < self.Thdis] = self.gbeta(alpha[a,:][dis[a,:] < self.Thdis], beta[a,:][dis[a,:] < self.Thdis]) * X[a,:][dis[a,:] < self.Thdis]
        #print(dbrssi)
        
        
        
        Gain = np.zeros(self.num_samples * self.num_sectors).reshape(self.num_samples, -1)
        Gain[a,:] = self.gain(alpha[a,:], beta[a,:])
        Gain[dis > self.Thdis] = 0.0
        #Gain[a,:][dis[a,:] < self.Thdis] = self.gain(alpha[a,:][dis[a,:] < self.Thdis], beta[a,:][dis[a,:] < self.Thdis])
        #print(Gain[a,:][dis < self.Thdis])
        rssi = self.tPow + Gain - self.pathloss
        rssi[rssi < -500] = -120
        #print(rssi)
        #print(np.sum(np.exp((rssi + self.fNum) * self.eNum), axis = 1)[-20:])
        rsrp = np.log(np.sum(np.exp((rssi + self.fNum) * self.eNum), axis = 1)) / self.eNum - self.fNum
        #print(rsrp)
        sinr = 10 * np.log(10 ** (rsrp / 10) / (10 ** (self.noise / 10) - 10 ** (rsrp / 10) + np.sum(10 ** (rssi / 10), axis = 1)))
        
        
        darsrp = np.zeros(self.num_samples).reshape(self.num_samples, -1)
        dbrsrp = np.zeros(self.num_samples).reshape(self.num_samples, -1)
        drsrppart1 = 1 / (np.sum(np.exp((rssi + self.fNum) * self.eNum), axis = 1)) * np.sum(np.exp(self.eNum * (self.fNum + rssi)), axis = 1)
        darsrp = drsrppart1.reshape(-1, 1) * darssi
        dbrsrp = drsrppart1.reshape(-1, 1) * dbrssi
        #print(darsrp)
        #print(dbrsrp)

        dasinr = np.zeros(self.num_samples).reshape(self.num_samples, -1)
        dbsinr = np.zeros(self.num_samples).reshape(self.num_samples, -1)        
        dsinrpart1 = 10 ** (self.noise / 10) + np.sum(10 ** (rssi / 10), axis =1).reshape(-1, 1)
        dsinrpart2 = np.sum(10 ** (rssi / 10), axis = 1).reshape(-1, 1)
        dsinrpart4 = 10 ** (self.noise / 10) - 10 ** (rsrp / 10).reshape(-1, 1)  
        dsinrpart5 = np.sum(10 ** (rssi / 10), axis = 1).reshape(-1, 1)
        dsinrpart3 = dsinrpart4 + dsinrpart5
        #print(darsrp.shape,dsinrpart1.shape,dsinrpart2.shape,darssi.shape,dsinrpart4.shape)
        dasinr = (darsrp * dsinrpart1 - dsinrpart2 * darssi) / dsinrpart3
        dbsinr = (dbrsrp * dsinrpart1 - dsinrpart2 * dbrssi) / dsinrpart3
        #print(dasinr)
        #print(dbsinr)
        
        coverrsrp = self.sigm(rsrp - self.rTh).reshape(-1, 1)
        #print(np.exp(self.rTh - rsrp))
        dcoverrsrppart1 = np.exp(self.rTh - rsrp) / ((1 + np.exp(self.rTh - rsrp)) ** 2)
        dcoverrsrppart1 = dcoverrsrppart1.reshape(-1, 1)
        dacoverrsrp = dcoverrsrppart1 * darsrp
        dbcoverrsrp = dcoverrsrppart1 * dbrsrp
        
        coversinr = self.sigm(sinr - self.sTh).reshape(-1, 1)
        dcoversinrpart1 = np.exp(self.sTh - sinr) / ((1 + np.exp(self.sTh - sinr)) ** 2)
        dcoversinrpart1 = dcoversinrpart1.reshape(-1, 1)
        dacoversinr = dcoversinrpart1 * dasinr
        dbcoversinr = dcoversinrpart1 * dbsinr
        
        coverpoint = coverrsrp * coversinr
        #print(coversinr.shape, dacoverrsrp.shape, coverrsrp.shape, dacoversinr.shape)
        dacoverpoint = coversinr * dacoverrsrp + coverrsrp * dacoversinr
        dbcoverpoint = coversinr * dbcoverrsrp + coverrsrp * dbcoversinr
        
        coverarea = np.sum(coverpoint) / self.num_samples
        dacoverarea = np.sum(dacoverpoint, axis = 0) / self.num_samples
        dbcoverarea = np.sum(dbcoverpoint, axis = 0) / self.num_samples
        return coverarea ,dacoverarea, dbcoverarea
    
    def dabs(self, x):#abs的导数
        x[x < 0] = -1
        x[x > 0] = 1
        x[x == 0] = 0
        return x
            
    def galpha(self, alpha,beta):
        return 0.0345045 * cos(1.001 * alpha) + 0.334935 * cos(2.002 * alpha) + \
             0.687086 * cos(3.003 * alpha) + 0.111471 * cos(4.004 * alpha) - \
             1.22372 * cos(5.005 * alpha) - 0.556876 * cos(6.006 * alpha) + \
             0.906706 * cos(7.007 * alpha) - 9.62862 * sin(1.001 * alpha) - \
             46.4864 * sin(2.002 * alpha) - 63.5435 * sin(3.003 * alpha) - \
             7.7117 * sin(4.004 * alpha) + 67.9178 * sin(5.005 * alpha) + \
             25.7357 * sin(6.006 * alpha) - 35.9459 * sin(7.007 * alpha) + \
             1/pi * (60.19 + 4.888 * cos(0.9274 * beta) - 22.29 * cos(1.8548 * beta) - \
                34.57 * cos(2.7822 * beta) - 13.96 * cos(3.7096 * beta) + \
                4.296 * cos(4.637 * beta) + 2.091 * cos(5.5644 * beta) - \
                0.6267 * cos(6.4918 * beta) - 2.317 * sin(0.9274 * beta) + \
                10.64 * sin(1.8548 * beta) + 27.56 * sin(2.7822 * beta) + \
                19.88 * sin(3.7096 * beta) - 9.596 * sin(4.637 * beta) - \
                9.043 * sin(5.5644 * beta) - 11.25 * sin(6.4918 * beta)) * self.dabs(alpha) - \
             1/pi * (15.5111 + 4.888 * cos(0.9274 * (-beta + pi)) - \
                22.29 * cos(1.8548 * (-beta + pi)) - \
                34.57 * cos(2.7822 * (-beta + pi)) - \
                13.96 * cos(3.7096 * (-beta + pi)) + \
                4.296 * cos(4.637 * (-beta + pi)) + \
                2.091 * cos(5.5644 * (-beta + pi)) - \
                0.6267 * cos(6.4918 * (-beta + pi)) - \
                2.317 * sin(0.9274 * (-beta + pi)) + \
                10.64 * sin(1.8548 * (-beta + pi)) + \
                27.56 * sin(2.7822 * (-beta + pi)) + \
                19.88 * sin(3.7096 * (-beta + pi)) - \
                9.596 * sin(4.637 * (-beta + pi)) - \
                9.043 * sin(5.5644 * (-beta + pi)) - \
                11.25 * sin(6.4918 * (-beta + pi))) * self.dabs(alpha)
        
        
    def gbeta(self, alpha,beta):
        return -(1 - abs(alpha) / pi) * (-2.14879 * cos(0.9274 * beta) + \
            19.7351 * cos(1.8548 * beta) + 76.6774 * cos(2.7822 * beta) + \
            73.7468 * cos(3.7096 * beta) - 44.4967 * cos(4.637 * beta) - \
            50.3189 * cos(5.5644 * beta) - 73.0328 * cos(6.4918 * beta) - \
            4.53313 * sin(0.9274 * beta) + 41.3435 * sin(1.8548 * beta) + \
            96.1807 * sin(2.7822 * beta) + 51.786 * sin(3.7096 * beta) - \
            19.9206 * sin(4.637 * beta) - 11.6352 * sin(5.5644 * beta) + \
            4.06841 * sin(6.4918 * beta)) - \
            1/pi * abs(alpha) * (2.14879 * cos(0.9274 * (-beta + pi)) - \
             19.7351 * cos(1.8548 * (-beta + pi)) - \
             76.6774 * cos(2.7822 * (-beta + pi)) - \
             73.7468 * cos(3.7096 * (-beta + pi)) + \
             44.4967 * cos(4.637 * (-beta + pi)) + \
             50.3189 * cos(5.5644 * (-beta + pi)) + \
             73.0328 * cos(6.4918 * (-beta + pi)) + \
             4.53313 * sin(0.9274 * (-beta + pi)) - \
             41.3435 * sin(1.8548 * (-beta + pi)) - \
             96.1807 * sin(2.7822 * (-beta + pi)) - \
             51.786 * sin(3.7096 * (-beta + pi)) + \
             19.9206 * sin(4.637 * (-beta + pi)) + \
             11.6352 * sin(5.5644 * (-beta + pi)) - \
             4.06841 * sin(6.4918 * (-beta + pi)))
     
     
    def gain(self, alpha, beta):
        #a = alpha
        #print(a)
        #print(self.f1(alpha).shape)
        return self.f1(alpha) - (np.abs(alpha) /pi) * (self.f1(pi) - self.f2(np.abs(alpha) / pi) * (self.f1(0) - self.f2(beta)))
    
    def sigm(self, X):
        return 1 / (1 + np.exp(-X))
        
    def f1(self, x):
        a0 =      -25.32  #(-25.79, -24.84)
        a1 =       9.619  #(9.039, 10.2)
        b1 =     0.03447  #(-0.7705, 0.8395)
        a2 =       23.22  #(22.25, 24.19)
        b2 =      0.1673  #(-0.9882, 1.323)
        a3 =       21.16  #(20.63, 21.69)
        b3 =      0.2288  #(-1.234, 1.692)
        a4 =       1.926  #(0.902, 2.949)
        b4 =     0.02784  #(-0.5237, 0.5793)
        a5 =      -13.57  #(-14.34, -12.79)
        b5 =     -0.2445  #(-1.771, 1.282)
        a6 =      -4.285  #(-4.678, -3.893)
        b6 =    -0.09272  #(-0.7394, 0.554)
        a7 =        5.13  #(4.747, 5.513)
        b7 =      0.1294  #(-0.7025, 0.9614)
        w =       1.001   #(0.994, 1.008)
        #print(x)
        #print(np.cos(x))
        return  a0 + a1*cos(x*w) + b1*sin(x*w) + \
               a2*cos(2*x*w) + b2*sin(2*x*w) + a3*cos(3*x*w) + b3*sin(3*x*w) + \
               a4*cos(4*x*w) + b4*sin(4*x*w) + a5*cos(5*x*w) + b5*sin(5*x*w) + \
               a6*cos(6*x*w) + b6*sin(6*x*w) + a7*cos(7*x*w) + b7*sin(7*x*w)
            
    def f2(self, x):
        a0 =      -42.31  #(-69, -15.61)
        a1 =      -4.888  #(-29.93, 20.16)
        b1 =       2.317  #(-11.76, 16.4)
        a2 =       22.29  #(8.05, 36.53)
        b2 =      -10.64  #(-22.61, 1.332)
        a3 =       34.57  #(-7.91, 77.06)
        b3 =      -27.56  #(-33.4, -21.72)
        a4 =       13.96  #(-20.21, 48.14)
        b4 =      -19.88  #(-31.63, -8.13)
        a5 =      -4.296  #(-13.22, 4.626)
        b5 =       9.596  #(1.89, 17.3)
        a6 =      -2.091  #(-13.97, 9.788)
        b6 =       9.043  #(5.628, 12.46)
        a7 =      0.6267  #(-15.67, 16.93)
        b7 =       11.25  #(4.521, 17.98)
        w =      0.9274   #(0.8637, 0.9911)
        return a0 + a1*cos(x*w) + b1*sin(x*w) + \
            a2*cos(2*x*w) + b2*sin(2*x*w) + a3*cos(3*x*w) + b3*sin(3*x*w) + \
            a4*cos(4*x*w) + b4*sin(4*x*w) + a5*cos(5*x*w) + b5*sin(5*x*w) + \
            a6*cos(6*x*w) + b6*sin(6*x*w) + a7*cos(7*x*w) + b7*sin(7*x*w)
        
    def main(self, theta):
        
        a = np.array([x for x in range(self.num_samples)])
        X = np.ones(self.dis.shape)
        for i in range(100):
            coverarea, dacoverarea, dbcoverarea = self.getrsri(X, a)#X是一组mini-batch
            self.dalpha[i] = dacoverarea
            self.dbeta[i] = dbcoverarea
            self.alpha1 += theta * dacoverarea
            self.beta1 += theta * dbcoverarea
            self.alpha = self.alpha2 - self.alpha1
            self.beta = self.beta1 - self.beta2
            #print(dacoverarea)
            print(i,"coverarea is %f" %coverarea)
        return self.alpha1, self.beta1, coverarea
        
        '''
        f = open(r"E:\2017.10.01.CMDI\2017.10.01.CMDI\coverrecord.txt", "w+")
        shuffleid = [x for x in range(self.num_samples)]#随机采样点ID
        random.shuffle(shuffleid)#随机采样点ID
        #print shuffleid
        for num,i in enumerate(shuffleid):
            X = np.ones(self.dis.shape)
            a = np.array([i])
            coverarea, dacoverarea, dbcoverarea = self.getrsri(X, a)#X是一组mini-batch
            
            self.dalpha[i] = dacoverarea
            self.dbeta[i] = dbcoverarea
            self.alpha1 += theta * dacoverarea
            self.beta1 += theta * dbcoverarea
            self.alpha = self.alpha2 - self.alpha1
            self.beta = self.beta1 - self.beta2
            f.write("%d, coverarea is %f"%(num, coverarea))
            f.write("\n")
            print(num,"coverarea is %f" %coverarea)
            
        f.close()
        return self.alpha1, self.beta1, coverarea
        '''
