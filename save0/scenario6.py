#!python2
#coding=utf-8

"""
多基站覆盖率研究
"""

from __future__ import print_function
from math import log10, pi, sqrt, atan2, atan, exp, pow, fmod
import cmath
import numpy as np
from collections import namedtuple
import simplejson as json
import codecs
import time
import copy
from scipy import spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon, Wedge, Rectangle

from antenna import gain, hv, vv
from pro_hgt import get_elevation
import rsrpsinr as rs
from rsrpsinr import degree_to_rad, rad_to_degree
import pprint
pp = pprint.PrettyPrinter(indent=4)

TERM_HEIGHT = 1.5
TX_POWER = 18.2
FREQ = 900
NOISE = -110.0
center_x = 103.75  # 兰州的中心点（保留2位小数）
center_y =  36.09

def get_elev(x, y):
    lon = x / 89.957 + center_x
    lat = y / 110.574 + center_y
    return get_elevation(lon, lat)

def map(lon, lat):
    new_x = (lon - center_x) * 89.957
    new_y = (lat - center_y) * 110.574
    return new_x, new_y

#####################################################################
# 研究x0,y0南北东西各2km的范围，大约有72个基站扇区。

RG =  2.0
x0 = 11.0
y0 = -3.0

lon1, lon2 = (x0-RG)/ 89.957 + center_x, (x0+RG)/ 89.957 + center_x
lat1, lat2 = (y0-RG)/110.574 + center_y, (y0+RG)/110.574 + center_y
print("BBox", lon1, lon2, lat1, lat2)

SAMPLES_HV = 100

def make_samples():
    samples = []
    for x in np.linspace(x0-2, x0+2., SAMPLES_HV):
        for y in np.linspace(y0-2, y0+2., SAMPLES_HV): ## 2km
            e = get_elev(x, y)
            samples.append( (x, y, e) )
    return np.array(samples, dtype=np.float64, order='C')

with codecs.open('basestation.json', 'r', "utf-8") as f:
    BList_ori = json.load(f)
    BList = []
    for id, b in enumerate(BList_ori):
        if -2 < b["x"]-x0 < 2 and -2 < b["y"]-y0 < 2:
            b["power"] = TX_POWER
            b["azimuth"] = degree_to_rad(b["azimuth"])
            b["mtilt"] = degree_to_rad(b["mtilt"])
            b["id"] = id
            BList.append(b)
    print("[Init] Read Base Station List. Num=", len(BList_ori), "of", len(BList) )

#BList = BList[:2] # to test only one basestation

#pp.pprint(BList)

def makeBaseStations(BList):
    base_id = []
    base_name = []
    out = []
    for b in BList:
        out.append(
            [
                b["x"], b["y"], 
                get_elev(b["x"], b["y"]),
                b["height"],
                b["azimuth"],
                b["mtilt"],
                b["power"]
            ]
        )
        base_id.append(b["id"])
        base_name.append(b["name"])
    return np.array(out, dtype=np.float64, order='C'), base_name, base_id
    
S = make_samples()
B, B_name, B_id = makeBaseStations(BList)  # B, BList: both in rad, not in degree.

print("Baselist B shape:", B.shape)
max_distance = 2.0
t0 = time.clock() 
tree = spatial.KDTree(B[:,:2])
t1 = time.clock() - t0
B_of_S =  tree.query_ball_point(S[:,:2], max_distance)
t2 = time.clock() - t0 - t1
print(type(B_of_S[0]), B_of_S.shape, B_of_S.dtype )
print(t1, "seconds process time: KD tree making")
print(t2, "seconds process time: KD tree query")
print("total samples", S.shape)

rs.set_antenna_hv_vv(hv, vv)

def drawBase(b, ax, text=""):
    x = b[0]
    y = b[1]
    
    if not text:
        text = str(text) 
    z = complex(x, y)
    r = 0.1

    a = b[4]
    t = rad_to_degree( b[5] )
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
        ax.text(zt.real, zt.imag, 
            #text,
            "%.1f" % t,
            ha='center', va='center', 
            #alpha=0.8,
            fontsize=6,
            fontproperties='FangSong')
    
    if -20.0 < t < 20.0:
        w = Polygon(poly, facecolor='yellow', edgecolor='navy', alpha=0.7, linewidth=1)
    else:
        w = Polygon(poly, facecolor='red', edgecolor='navy', alpha=0.7, linewidth=1)
    ax.add_patch(w)

if __name__ == "__main__":

    # Visualization
    IMG_FILENAME = "lanzhou_study.png"
    IMG_JSON = IMG_FILENAME + ".json"
    with open(IMG_JSON, 'r') as f:
        img_info = json.load(f)
    bb = img_info["bbox"]
    ibb0, ibb1 = map(bb[0], bb[1])
    ibb2, ibb3 = map(bb[2], bb[3])
    ibb = ibb0, ibb1, ibb2, ibb3

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    img=plt.imread(IMG_FILENAME)
    plt.imshow(img, interpolation='lanczos', origin='upper', 
        extent=[ibb0, ibb2, ibb1, ibb3],
        alpha=.5
        )

    from scipy.optimize import minimize
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
        covered_s_count, isCovered, bIndex, rsrp, sinr, noise = rs.get_coverage(S, local_B, B_of_S)
        if iters%50==0:
            print("covered_s_count ", covered_s_count, "at:", iters)
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
        
    if False:   
        t0 = time.clock()
        res = minimize(f, x0, args, 
            method='Powell',
            #method='Nelder-Mead',
            options={'xtol': 1e-3, 
                    'disp': True,
                    'maxiter':2, # for powell, 1900 fev at 1 iter; 4967 at 2 iter
                    'maxfev':2000,
                    },
            callback=callme)
        t1 = time.clock() - t0
    
        pp.pprint(res)
    
        import sys
        sys.exit()

    opt_x  = np.array([ 
         9.03949388e+00,   6.99068537e-02,   2.32266463e+00,
         1.22689280e+00,  -3.48312078e-01,   6.28654188e-02,
         9.30166667e-01,   5.54689145e-02,   2.73557889e+00,
         2.95812534e-02,   8.17018742e+00,   9.13130289e-01,
        -8.58238115e-01,   6.98927438e-02,   6.17590957e+00,
         8.56964470e-02,  -1.32526781e-01,   7.60856213e-02,
         1.47773032e+00,   3.76322207e-02,   8.86087459e-01,
         2.34455359e-02,   1.88939109e+00,   1.82747350e+00,
        -6.79837372e-01,   9.95709384e-03,   4.33046307e+00,
         5.30789158e+00,   3.11809492e+00,   6.70202074e-02,
         5.81069182e+00,   2.64376538e+00,   2.82154408e+00,
         1.85145139e-02,   5.83680778e+00,  -1.55146196e+00,
         3.98310433e+00,   8.69655003e-01,   1.06766456e+00,
         4.46262485e-02,   4.43703492e+00,   4.57685569e+00,
         2.03853744e+00,   2.66499163e-02,   3.09149269e+00,
         4.32894237e-02,   4.48671754e+00,   2.11873963e-02,
         1.44095894e+00,   6.27428198e-02,   7.60806880e-01,
        -2.19473950e-02,   2.63229128e+00,   1.87805420e+00,
         6.54300629e+00,   5.36505576e-02,   2.25528040e+00,
         2.28029833e-02,   4.12648431e+00,   8.55879617e-02,
         4.74779117e+00,   4.99208255e-02,   5.67700055e+00,
         8.11574133e-02,   4.34969200e+00,   7.96021241e-02,
        -1.75483307e-01,   4.81995153e-02,   5.69813250e+00,
         3.71316500e-02,   6.08717315e+00,   1.15080793e+00,
         3.62610999e+00,   3.50740634e-02,   2.10365640e+00,
         9.98482594e-02,   4.10772815e+00,   6.10352453e-02,
         4.36851173e+00,   3.84741266e-02,   1.07237698e+01,
         2.86528354e+00,   4.32407112e+00,   6.10446941e-02,
         3.16670531e+00,   7.55185910e-02,   6.24032690e+00,
         1.11912047e-01,   7.40380891e+00,   2.95516749e-02,
         8.89830272e+00,   2.30115738e+00,   3.44982921e+00,
         6.91077508e-02,   5.66147122e-02,   1.22312811e-02,
         7.40293155e-01,   1.07568723e+00,   3.28774858e+00,
        -5.10714951e-03,   3.74314859e+00,   6.08255453e-02,
        -7.19540990e-01,   8.90816214e-02,  -7.49474820e-02,
         3.68940122e-01,   3.48553662e+00,   3.02217303e+00,
         1.51615786e+00,   9.62020737e-02,   1.76644918e+00,
         2.26892803e-01,   6.50911751e+00,   3.46380226e-02,
         2.74891164e+00,   3.03691701e+00,   1.96068089e+00,
         1.36682142e-01,   1.01551440e+00,   1.38352249e-02,
         1.06062038e+00,   1.03515989e-01,   4.18828545e+00,
         9.64666518e-02,   1.61668844e+00,   4.56295936e-02,
         7.66332314e+00,   2.59585692e+00,   1.89758003e+00,
        -3.17264619e-03,   4.47090987e+00,   3.73576603e-02,
         2.00739557e+00,   1.55657124e-01,   3.36328070e+00,
         1.36109725e-01,   4.62123225e+00,   5.92786256e-03,
         8.24015047e+00,   1.64111889e-01,   4.41612558e+00,
         8.83076095e-02,   6.05641418e+00,   1.18276502e-01
    ])


    B[:,4] = np.fmod( opt_x[::2], 2*pi) 
    B[:,5] = np.fmod( opt_x[1::2], 2*pi) 
    for b in B:
        print("%.1f, %.1f" % (rad_to_degree(b[4]), rad_to_degree(b[5]) ))
    t0 = time.clock()
    covered_s_count, isCovered, bIndex, rsrp, sinr, noise = rs.get_coverage(S, B, B_of_S)
    t1 = time.clock() - t0
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
        drawBase(b, plt.gca(), B_id[nid])
        #drawBase(b, plt.gca(), B_name[nid])

    plt.savefig('6-cover10000_opt.png', bbox_inches='tight', dpi=180)
    plt.close()
    #plt.show()


    print("    Covered: ", covered_s_count)
    print(t1, "seconds process time")
