#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
from xml.dom.minidom import parse
import xml.dom.minidom
 
# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse("Paramster.xml")
collection = DOMTree.documentElement
if collection.hasAttribute("shelf"):
   print ("Root element : %s" % collection.getAttribute("shelf"))
 
# 在集合中获取所有电影
rows = collection.getElementsByTagName("ROW")
rowsall = [] 
# 打印每部电影的详细信息
for row in rows:
   AntennaHeight = row.getElementsByTagName('Column0')[0]
   AntennaDirection = row.getElementsByTagName('Column1')[0]
   AntennaMechanicalDownTilt = row.getElementsByTagName('Column2')[0]
   UserNodeBID = row.getElementsByTagName('Column5')[0] 
   Longitude = row.getElementsByTagName('Column6')[0]
   Latitude = row.getElementsByTagName('Column7')[0]
   rowall =  [int(AntennaHeight), int(AntennaDirection), int(AntennaMechanicalDownTilt), int(UserNodeBID), float(Longitude), float(Latitude)]
   rowsall.append(rowall)
print(rowsall)
