#!python2
#coding=utf-8

"""
兰州市的覆盖率分析
"""

from __future__ import print_function
import sys
from math import log10, pi, sqrt, atan2, atan, exp, pow
import cmath
import copy
import numpy as np
from collections import namedtuple
import simplejson as json
import codecs
import time
from scipy import spatial
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon, Wedge, Rectangle

from antenna import gain, hv, vv
import rsrpsinr as rs
from rsrpsinr import degree_to_rad, rad_to_degree

TERM_HEIGHT = 1.5
TX_POWER = 18.2 # dB
FREQ = 900
NOISE = -110.0
center_x = 103.75  # 兰州的中心点（保留2位小数）
center_y =  36.09

#####################################################################
"""
取得所有的env中的点作为采样点，约72万
"""
def make_samples():
    t1 = time.time()
    samples = []
    with codecs.open('env.json', 'r', "utf-8") as f:
        env = json.load(f)
    for b in env:
        x = b["x"]
        y = b["y"]
        e = b["height"]
        samples.append( (x, y, e) )
    delay = time.time()-t1
    #print("Init samples in %.2fs" % delay)
    return np.array(samples, dtype=float, order='C')

"""
取得所有的扇区，约600个
"""
def makeBaseStations(samples):

    t1 = time.time()

    # 做一棵树，准备查询基站的海拔
    tree_samples = spatial.cKDTree(samples[:,:2])
    # 读入基站位置
    with codecs.open('basestation.json', 'r', "utf-8") as f:
        BList_ori = json.load(f)
    BList = []
    for id, b in enumerate(BList_ori):
        b["power"] = TX_POWER
        b["azimuth"] = degree_to_rad(b["azimuth"])
        b["mtilt"] = degree_to_rad(b["mtilt"])
        b["id"] = id
        BList.append(b)

    
    #print("[Init] Read Base Station List. Num=", len(BList) )
    
    base_id = []
    base_name = []
    out = []
    for b in BList:
        x, y = b["x"], b["y"]
        dist, id_of_sample = tree_samples.query((x,y))
        assert dist < 0.030 # 30m
        e = samples[id_of_sample][2]
        #print("dist=", dist, "e=", e)
        out.append(
            [
                b["x"], b["y"], 
                e,
                b["height"],
                b["azimuth"],
                b["mtilt"],
                b["power"]
            ]
        )
        base_id.append(b["id"])
        base_name.append(b["name"])
    delay = time.time()-t1
    #print("Init baseStation in %.2fs" % delay)
    return np.array(out, dtype=float, order='C'), base_name, base_id


def drawBase(b, ax, text=""):
    x = b[0]
    y = b[1]
    
    if not text:
        text = str(text) 
    z = complex(x, y)
    r = 0.1

    a = b[4]
    za = cmath.rect(r, pi/2.-a) # 朝向角度
    z0 = z + za                 # 位置
    zr = cmath.rect(1, pi/16.)  
    z1 = z + za * zr
    z2 = z + za * zr * zr
    z_1 = z + za / zr
    z_2 = z + za / zr / zr
    poly = [(x,y)]
    for z in [z_2, z_1, z0, z1, z2]:
        poly.append((z.real, z.imag))
    if text:
        zt = z + za * 1.5
        ax.text(zt.real, zt.imag, text, ha='center', va='center', 
            #alpha=0.8,
            fontsize=4,
            fontproperties='FangSong')
    w = Polygon(poly, facecolor='yellow', edgecolor='navy', alpha=0.7, linewidth=1)
    ax.add_patch(w)

def getdata():
    S = make_samples()
    B, B_name, B_id = makeBaseStations(S)  # B, BList: both in rad, not in degree.
    B_of_S = np.array([[i for i in range(len(B))]] * len(S),dtype = np.float64)
    #print (B_of_S.shape,"B_of_S.shape")
    #print(B.shape,"B.shape")
    #print(S.shape,"S.shape")
    rs.set_antenna_hv_vv(hv, vv)
    Pathloss, dis, Alpha, Beta, covered_s_count = rs.get_coverage(S, B, B_of_S)
    #Pathloss, dis, Alpha, Beta = rs.get_coverage(S, B, B_of_S)
    #print(Pathloss.T.shape)
    #print(dis.T.shape)
    #print(Pathloss.T[:5,:])
    #print(dis.T[:5,:])
    #print(Alpha.shape)
    #print(Pathloss.shape,dis.shape,Alpha.shape,Beta.shape)
    return Pathloss, dis, np.array(list(B[:,4])*S.shape[0]).reshape(S.shape[0], -1), np.array(list(B[:,5])*S.shape[0]).reshape(S.shape[0], -1), np.array(list(B[:,6])*S.shape[0]).reshape(S.shape[0], -1), Alpha, Beta, covered_s_count
    #return Pathloss, dis, np.array(list(B[:,4])*S.shape[0]).reshape(S.shape[0], -1), np.array(list(B[:,5])*S.shape[0]).reshape(S.shape[0], -1), np.array(list(B[:,6])*S.shape[0]).reshape(S.shape[0], -1), Alpha, Beta


def writetxt(path, data):
    f = open(r"E:\2017.10.01.CMDI\2017.10.01.CMDI\%s.txt"%path, "w+")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write(str(data[i,j]))
            f.write("\t")
        f.write("\n")
    f.close()



if __name__ == "__main__":

    pathloss,dis,alpha, beta, tpow = getdata()
    '''
    writetxt("pathloss",pathloss)
    writetxt("dis",dis)
    writetxt("alpha",alpha)
    writetxt("beta",beta)
    writetxt("tpow",tpow)
    '''
