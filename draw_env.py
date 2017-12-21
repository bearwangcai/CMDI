#!python2
#coding=utf-8
from __future__ import print_function

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

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
   
    colormap = {
    u'Wet_land': "navy",
    u'Green_Land': "yellowgreen",
    u'Irregular_Large_Building':"black",
    u'Ordinary_Building':"black",
    u'Urban_Open_Area':"yellow",
    u'Hight_Building':"black",
    u'Parallel_Regular_Building':"black",
    u'no value':"white",
    u'Water':"blue",
    u'Irregular_Building':"black",
    u'Forest':"green",
    u'Suburban_Village':'grey',
    u'Sea':'darkblue',
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

    ax.scatter(xx, yy, c=cc, edgecolor='none', marker='x', alpha=0.3, s=3 )
    plt.xlabel("distance(km)")
    plt.ylabel("distance(km)")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig('env-basemap.png', bbox_inches='tight', dpi=200)
    plt.close()
    #plt.show()

drawEnv()
    