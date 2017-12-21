#!python2
#coding=utf-8

"""
地形地貌数据的预处理
后文将以公里为单位展开研究，因此把经纬度转为经纬方向的公里。
数据存储在json格式中。
"""

from __future__ import print_function, unicode_literals

import codecs
import simplejson as json

center_x = 103.75  # 兰州的中心点（保留2位小数）
center_y =  36.09

def map(lon, lat):
    new_x = (lon - center_x) * 89.957
    new_y = (lat - center_y) * 110.574
    return new_x, new_y

def save_env_json():
    """
    Read from txt
    """
    kind_set = set()
    bb = []
    with open("heightInfo-Lanzhou.txt", "rt") as f:
        for line in f:
            if len(line.strip())==0:
                continue
            records = line.split("\t")
            lon = float(records[0])
            lat = float(records[1])
            height = float(records[2])
            kind = records[3].strip()
            kind_set.add(kind)
            x1, y1 = lon, lat
            #x1, y1 = lon, lat
            xm, ym = map(x1, y1)
            bs = { 
                "x": xm,
                "y": ym,
                "lon": x1,
                "lat": y1,
                "kind": kind,
                "height":height,
            }   
            bb.append(bs)
    with codecs.open('env.json', 'w', "utf-8") as f:
        json.dump(bb, f, ensure_ascii=False, sort_keys=True, indent=4 * ' ')
    print( kind_set )
    ##u'Wet_land', u'Green_Land', u'Irregular_Large_Building', u'Ordinary_Building', u'Urban_Open_Area', u'Hight_Building', u'Parallel_Regular_Building', u'no value', u'Water', u'Irregular_Building', u'Forest'

save_env_json()
