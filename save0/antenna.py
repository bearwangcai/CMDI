#!python2
#coding=utf-8
"""
天线资源的读取
输出 hv,vv numpy数组
输出 gain函数
"""

from __future__ import print_function

import numpy as np
from math import pi

#ANTENNA_FILENAME = "TY2600.txt"
ANTENNA_FILENAME = "GSM900.txt"

def init():
    global hv, vv
   
    f = open(ANTENNA_FILENAME, "rt")
    text = f.readlines()
    
    assert text[0].strip()=="Gain"
    gain = float(text[1])
    
    LN = 2
    assert text[LN].startswith("HORIZONTAL")
    hv = []
    for i in range(360):
        a, v = text[LN+1+i].split("\t")  ## 角度/增益
        assert int(a) == i
        hv.append(float(v))
    hv.append(hv[0])  ## 0度=360度
    
    LN = 363
    assert text[LN].startswith("VERTICAL")
    vv = []
    for i in range(360):
        a, v = text[LN+1+i].split("\t")  ## 角度/增益
        assert int(a) == i
        vv.append(float(v))
    vv.append(vv[0])
    
    hv = np.asarray(hv)
    vv = np.asarray(vv)
    f.close()
    
def antenna_h_rad(angle):
    """
    @param angle: 0~2$\pi$
    @return dB
    """
    d = angle/pi*180
    return antenna_h_deg(d)

def antenna_h_deg(angle):
    """
    @param angle: 0~360
    @return dB
    """
    angle -= (angle//360) * 360
    i = int(angle)
    f = angle - i
    return hv[i]*(1-f) + hv[i+1]*f

def antenna_v_rad(angle):
    """
    @param angle: 0~2$\pi$
    @return dB
    """
    d = angle/pi*180
    return antenna_v_deg(d)

def antenna_v_deg(angle):
    """
    @param angle: 0~360
    @return dB
    """
    angle -= (angle//360) * 360
    i = int(angle)
    f = angle - i
    return vv[i]*(1-f) + vv[i+1]*f
    
def gain(alpha, beta):
    ## alpha --> [-pi, pi]
    alpha -= (alpha//(2*pi)) * 2 * pi
    if alpha > pi:
        alpha = alpha - 2 * pi
    f  = abs(alpha) / pi
    C1 = f * ( antenna_h_rad(pi) - antenna_v_rad(pi-beta) )
    C2 = (1-f) * ( antenna_h_rad(0) - antenna_v_rad(beta) )
    return antenna_h_rad(alpha) - C1 - C2

init()

if __name__ == "__main__":
    from math import cos, sin
    import matplotlib
    import matplotlib.pyplot as plt
    # Check all styles with print(plt.style.available)
    # plt.style.use(['seaborn-white', 'bmh'])  
    plt.style.use(['bmh'])  
    #plt.style.use(['ggplot', 'bmh'])
    matplotlib.rc('font',family='Times New Roman')
    matplotlib.rc('font',size=10.0)

    print("H(0) = %.2f" % antenna_h_rad(0) )
    print("H(pi) = %.2f" % antenna_h_rad(pi) )
    print("V(0) = %.2f" % antenna_v_rad(0) )
    print("V(pi) = %.2f" % antenna_v_rad(pi) )

    theta = np.linspace(-pi, pi, 360, endpoint=True) ## 绘图横坐标

    def f(x):
        """水平方向"""
        return gain(x, 0./180*pi)
    
    def g(x):
        """垂直方向"""
        return gain(0, x)
    
    
    if 1: 
        # 极坐标图
        fig, ax_list = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(6,3), dpi=200)
        ## Either ax_list.plot Or subplot(211) then plot

        ax_list[0].set_theta_zero_location('N')
        ax_list[0].set_theta_direction('clockwise')
        h = np.vectorize(f)(theta)
        ax_list[0].plot(theta, h, linewidth=1)
        ax_list[0].set_ylim(-30,20)
        ax_list[0].set_yticks(np.array([-20, -10, 0, 5, 10, 15 ]))
        ax_list[0].set_title("Horizon")
        
        ax_list[1].set_theta_zero_location('N')
        ax_list[1].set_theta_direction('clockwise')
        v = np.vectorize(g)(theta)
        ax_list[1].plot(theta, v, linewidth=1)
        ax_list[1].set_ylim(-30,20)
        ax_list[1].set_yticks(np.array([-20, -10, 0, 5, 10, 15]))
        ax_list[1].set_title("Vertical")

        fig.savefig("antenna.png")
        plt.close(fig)
        #plt.show()
    
    if 0:
        # 直角坐标系图
        plt.figure(figsize=(6,3), dpi=180)
        
        ax = plt.subplot(121)
        h = np.vectorize(f)(theta)
        ax.plot(theta, h, linewidth=1)
        ax.plot((-pi, pi), (gain(0,0)-3,gain(0,0)-3), 'k-') 
        ax.set_ylim(-40,20)
        ax.set_yticks(np.array([-40, -30, -20, -10, 0, 18]))
        ax.set_xticks( [-pi, -pi/2, 0, pi/2, pi])
        ax.set_xticklabels( ["-180", "-90", "0", "+90", "+180"])
        ax.grid(True, linestyle='-', linewidth='0.5', color='red')
        ax.set_title("Horizon")
        
        ax = plt.subplot(122)
        v = np.vectorize(g)(theta)
        ax.plot(theta, v, linewidth=1)
        ax.plot((-pi, pi), (gain(0,0)-3,gain(0,0)-3), 'k-') 
        ax.set_ylim(-40,20)
        ax.set_yticks(np.array([-40, -30, -20, -10, 0, 18]))
        ax.set_xticks( [-pi, -pi/2, 0, pi/2, pi])
        ax.set_xticklabels( ["-180", "-90", "0", "+90", "+180"])
        ax.grid(True, linestyle='-', linewidth='0.5', color='red')
        ax.set_title("Vertical")

        plt.savefig("antenna_linear.png")
        #plt.show()