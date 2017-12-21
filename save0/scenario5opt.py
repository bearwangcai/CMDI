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
    print("Init samples in %.2fs" % delay)
    return np.array(samples, dtype=np.float64, order='C')

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

    
    print("[Init] Read Base Station List. Num=", len(BList) )
    
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
    print("Init baseStation in %.2fs" % delay)
    return np.array(out, dtype=np.float64, order='C'), base_name, base_id


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

if __name__ == "__main__":

    S = make_samples()
    B, B_name, B_id = makeBaseStations(S)  # B, BList: both in rad, not in degree.

    print("Baselist B shape:", B.shape)

    t0 = time.clock() 
    tree = spatial.cKDTree(B[:,:2])
    t1 = time.clock() - t0
    max_distance = 1.0
    B_of_S =  tree.query_ball_point(S[:,:2], max_distance)
    t2 = time.clock() - t0 - t1
    print(type(B_of_S[0]), B_of_S.shape, B_of_S.dtype )

    print(t1, "seconds process time: KD tree making")
    print(t2, "seconds process time: KD tree query")
    
    print("total samples:", S.shape)

    rs.set_antenna_hv_vv(hv, vv)

    # Visualization
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)

    t0 = time.clock()
    covered_s_count, isCovered, bIndex, rsrp, sinr, noise = rs.get_coverage(S, B, B_of_S,
        RSRP_TH=-88,
        SINR_TH=-3,
        TERM_HEIGHT=1.5,
        FREQ=900,
        NOISE=-110
        )
    t1 = time.clock() - t0
    print(t1, "seconds process time in rs.get_coverage")
    my_cmap = mpl.colors.ListedColormap(['r', 'lightgreen'])

    print( S[:,0].shape )
    print( S[:,1].shape )
    ax.scatter(S[:,0], S[:,1], c=isCovered, 
        edgecolor='none', cmap=my_cmap,
        alpha=0.5, 
        marker='s',
        s=2 
        )
    plt.xlabel("distance(km)")
    plt.ylabel("distance(km)")
    print("Base:", B.shape)
    for nid, b in enumerate(B):
        drawBase(b, plt.gca()) # B_id[nid])
        #drawBase(b, plt.gca(), B_name[nid])
    
    ax.set_aspect('equal')
    plt.savefig('5-cover.png', bbox_inches='tight', dpi=200)
    plt.close()
    #plt.show()
    print("    Covered: ", covered_s_count)

    
    iters = 0
    def f(x, *args):
        """
        Parameter
        ========
        x: 待优化的变量，假设是每个扇区的azimuth和mtilt，呈现 a1,m1,a2,m2的形式；如果x包含高度、功率的信息也可以。
        args: [S, B, B_of_S]
        """
        global iters
        iters += 1
        local_B = copy.copy(B) # 此处对B根据x的参数进行各种修改，看看结果如何
        local_B[:,4] = x[::2]
        local_B[:,5] = x[1::2]
        covered_s_count, isCovered, bIndex, rsrp, sinr, noise = rs.get_coverage(
                S, local_B, B_of_S,
                RSRP_TH=-88,
                SINR_TH=-3,
                TERM_HEIGHT=1.5,
                FREQ=900,
                NOISE=-110)
        if iters%50==0:
            print("covered_s_count ", covered_s_count, "at:", iters)
            with codecs.open("r-%05d.json" % iters, 'w', "utf-8") as f:
                json.dump(list(x), f, ensure_ascii=False, sort_keys=True, indent=4 * ' ')
        return - 1.0 * covered_s_count

    
    x0 = []
    for b in B:
        x0.append( b[4])
        x0.append( b[5])
    x0 = np.asarray(x0)
    args = [S, B, B_of_S]

    iter_count = 0
    def callme(x):
        global iter_count
        iter_count+=1
        print("Min iter=", iter_count)
        
    if True:   
        t0 = time.clock()
        res = minimize(f, x0, args, 
            method='Powell',
            #method='Nelder-Mead',
            options={'xtol': 1e-3, 
                    'disp': True,
                    'maxiter':2, 
                    'maxfev':2000,
                    },
            callback=callme)
        t1 = time.clock() - t0
    
        pp.pprint(res)
    

        sys.exit()

