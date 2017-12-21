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

def readBaseStation():
    with codecs.open('basestation.json', 'r', "utf-8") as f:
        base = json.load(f)
    print("[Init] Read Base Station List.")
    print("Len(base)=", len(base))
    return base

def readEnv():
    """
    读入env.json文件
    """
    with codecs.open('env.json', 'r', "utf-8") as f:
        env = json.load(f)
    print("[Init] Read Env.")
    return env

IMG_FILENAME = "env-basemap-bg.png"
IMG_JSON = IMG_FILENAME + ".json"

def drawBaseStation():
    # Check all styles with print(plt.style.available)
    # plt.style.use(['seaborn-white', 'bmh'])  
    plt.style.use(['bmh'])  
    #plt.style.use(['ggplot', 'bmh'])
    mpl.rc('font',family='Times New Roman')
    mpl.rc('font',size=10.0)

    with open(IMG_JSON, 'r') as f:
        img_info = json.load(f)
    bb = img_info["bbox"]
    ibb0, ibb1 = bb[0], bb[1]
    ibb2, ibb3 = bb[2], bb[3]
    ibb = ibb0, ibb1, ibb2, ibb3
    print(IMG_JSON, ibb )

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    if False:
        img=plt.imread(IMG_FILENAME)
        plt.imshow(img, interpolation='lanczos', origin='upper', 
            extent=[ibb0, ibb2, ibb1, ibb3]
            )

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

    if True:
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
        ax.scatter(xx, yy, c=cc, edgecolor='none', marker='x', alpha=0.05, s=1 )

    plt.xlabel("distance(km)")
    plt.ylabel("distance(km)")
    
    base = readBaseStation()
    xx = []
    yy = []
    for b in base:
        x = b["x"]
        y = b["y"]
        xx.append(x)
        yy.append(y)

    ax.scatter(xx, yy, c='red', edgecolor='red', alpha=0.8, s=9 )
    
    ax = plt.gca()
    ax.set_aspect('equal')

    plt.savefig('ex-basemap.png', bbox_inches='tight', dpi=200)
    #plt.close()
    plt.show()

drawBaseStation()
    