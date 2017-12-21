#!python2
#coding=utf-8
from __future__ import print_function

"""
原计划存储一个确切大小的图片作为底图，但是目前的方式还是不准确。
"""




#from IPython import embed

import numpy as np
from collections import namedtuple
import codecs
import simplejson as json
import matplotlib as mpl
import matplotlib.pyplot as plt

def readEnv():
    """
    读入env.json文件
    """
    with codecs.open('env.json', 'r', "utf-8") as f:
        env = json.load(f)
    print("[Init] Read Env.")
    return env

center_x = 103.75  # 兰州的中心点（保留2位小数）
center_y =  36.09
def map(lon, lat):
    new_x = (lon - center_x) * 89.957
    new_y = (lat - center_y) * 110.574
    return new_x, new_y

def drawEnv():
    # Check all styles with print(plt.style.available)
    # plt.style.use(['seaborn-white', 'bmh'])  
    plt.style.use(['bmh'])  
    #plt.style.use(['ggplot', 'bmh'])
    mpl.rc('font',family='Times New Roman')
    mpl.rc('font',size=10.0)

    
   
    colormap = {
    u'Wet_land': "navy",
    u'Green_Land': "yellowgreen",
    u'Irregular_Large_Building':"silver",
    u'Ordinary_Building':"silver",
    u'Urban_Open_Area':"yellow",
    u'Hight_Building':"black",
    u'Parallel_Regular_Building':"silver",
    u'no value':"white",
    u'Water':"blue",
    u'Irregular_Building':"silver",
    u'Forest':"green",
    u'Suburban_Village':'grey',
    u'Sea':'seagreen',
    u'Suburban_Open_Area':'white'
    }

    env = readEnv()
    print("Size=", len(env))
    xx = []
    yy = []
    cc = []
    for b in env:
        x = b["x"]
        y = b["y"]
        c = colormap[b["kind"]]
        xx.append(x)
        yy.append(y)
        cc.append(c)

    xx = np.asarray(xx)
    yy = np.asarray(yy)
    bb = np.min(xx),np.min(yy),np.max(xx),np.max(yy)
    print("x-min y-min, x-max, y-max", bb)
    xsize = (bb[2]--bb[0])*10.0
    ysize = (bb[3]--bb[1])*10.0
    fig = plt.figure(figsize=(xsize, ysize),frameon=False)
    ax = plt.subplot(111)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim(bb[0],bb[2])
    ax.set_ylim(bb[1],bb[3])
    ax.scatter(xx, yy, c=cc, edgecolor='none', marker='x', alpha=0.3, s=3 )
    #plt.xlabel("distance(km)")
    #plt.ylabel("distance(km)")
    #ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig('env-basemap-bg.png', bbox_inches='tight', dpi=50)
    plt.close()
    #plt.show()

drawEnv()
    