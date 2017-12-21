#!python2
#coding=utf-8

"""
基站数据的预处理
后文将以公里为单位展开研究，因此把经纬度转为经纬方向的公里。
以兰州的bbox中心为原点。
zoom大概是12显示全城。
The approximate conversions are:
Latitude: 1 deg = 110.574 km
Longitude: 1 deg = 111.320*cos(latitude) km
因此兰州每经度 111.320*cos(36.09)=89.957 km

数据存储在json格式中。
"""

from __future__ import print_function

import cmath
import codecs
from collections import namedtuple
from math import atan2, log10, pi, sqrt

#import h5py
import numpy as np
import pandas as pd
import simplejson as json


BaseStationColumns = ['id', 'name', "longtitude", "latitude", "azimuth", "height", "mtilt"]

def degree_to_rad(d):
    return d*pi/180.0

def rad_to_degree(d):
    return d/pi*180.0

def init():
    """
    Read from xlsx
    """
    global baseStation, minx, miny, maxx, maxy
    df = pd.read_excel("basepart1.xlsx")#可更改项
    df.columns = BaseStationColumns
    baseStation = df
    print("[Init] Read Base Station List.")
    minx = min(baseStation["longtitude"])
    miny = min(baseStation["latitude"])
    maxx = max(baseStation["longtitude"])
    maxy = max(baseStation["latitude"])
    print("    in %.3f, %.3f, %.3f, %.3f" % (minx, miny, maxx, maxy))


# bbox = 103.536515833178,36.0133159544422,103.974174121672,36.173094526214
center_x = 103.75  # 兰州的中心点（保留2位小数）
center_y =  36.09

def map(lon, lat):
    new_x = (lon - center_x) * 89.957
    new_y = (lat - center_y) * 110.574
    return new_x, new_y

def write_json():
    BaseStationColumns = ['id', 'name', "x", "y", "azimuth", "height", "mtilt"]
    baseStation.columns = BaseStationColumns
    base = []
    for i in baseStation.index:
        b = baseStation.iloc[i]
        x = b["x"]
        y = b["y"]
        x1, y1 = x, y
        xm, ym = map(x1, y1)
        bs = { 
            "id": int(b["id"]),
            "name": b["name"],
            "x": xm,
            "y": ym,
            "lon": x,
            "lat": y,
            "azimuth": int(b["azimuth"]),  # np.int64!
            "height": float(b["height"]),
            "mtilt": float(b["mtilt"])
        }
        base.append(bs)
    with codecs.open('basestation.json', 'w', "utf-8") as f:
        json.dump(base, f, ensure_ascii=False, sort_keys=True, indent=4 * ' ')
    with codecs.open('basestation.txt', 'w', "utf-8") as f:
        for x in base:
            f.write(u"{id}\t{name}\t{lon}\t{lat}\t{azimuth}\t{height}\t{mtilt}\n".format(**x))

        
init()
write_json()
