﻿#!python
#cython: boundscheck=False, cdivision=True, wraparound=False
#coding=utf8 

import cython
cimport cython

import numpy as np
cimport numpy as np

import jinja2, codecs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


from libc.math cimport sin, cos, sqrt, fabs, atan, atan2, log, exp, log10, pow, floor, ceil 

cdef:
    double pi = 3.1415926535897932384626  # π
    double pi180 = pi/180.0
    DTYPE_t[:] ahv
    DTYPE_t[:] avv

def degree_to_rad(double d):
    return d*pi180

def rad_to_degree(double d):
    return d/pi180

cdef inline double cdegree_to_rad(double d):
    return d*pi180

cdef inline double crad_to_degree(double d):
    return d/pi180

def set_antenna_hv_vv(
    np.ndarray[DTYPE_t, mode='c', ndim=1] hv,
    np.ndarray[DTYPE_t, mode='c', ndim=1] vv):
    global ahv, avv
    ahv = hv #.copy()
    avv = vv #.copy()

cdef inline double antenna_h_rad(double angle):
    """
    @param angle: 0~2$\pi$
    @return dB
    """
    cdef double d = angle/pi180
    return antenna_h_deg(d)


cdef inline double antenna_h_deg(double angle):
    """
    @param angle: 0~360
    @return dB
    """
    global ahv 
    angle -= floor(angle/360.0) * 360.0
    cdef int i = <int>angle
    cdef double f = angle - i
    return ahv[i]*(1-f) + ahv[i+1]*f

cdef inline double antenna_v_rad(double angle):
    """
    @param angle: 0~2$\pi$
    @return dB
    """
    cdef double d = angle/pi*180
    return antenna_v_deg(d)


cdef inline double antenna_v_deg(double angle):
    """
    @param angle: 0~360
    @return dB
    """
    global avv
    angle -= floor(angle/360.0) * 360.0
    cdef int i = <int>angle
    cdef double f = angle - i
    return avv[i]*(1-f) + avv[i+1]*f
    
cdef inline double cgain(double alpha, double beta):
    ## alpha --> [-pi, pi]
    alpha -= floor(alpha/(2*pi)) * 2 * pi
    if alpha > pi:
        alpha = alpha - 2 * pi
    cdef double f  = abs(alpha) / pi
    cdef double C1 = f * ( antenna_h_rad(pi) - antenna_v_rad(pi-beta) )
    cdef double C2 = (1-f) * ( antenna_h_rad(0) - antenna_v_rad(beta) )
    return antenna_h_rad(alpha) - C1 - C2

cdef inline double loss(double d, double hB, double hR, double f, double C=3):
    """
    @param d: km
    @param f: MHz
    @param hB, hR: m
    @param C: C=3 for metropolitan areas
    """
    cdef double a = (1.1 * log10(f) - 0.7 ) * hR - (1.56 * log10(f) - 0.8)
    cdef double L = 46.3 + 33.9*log10(f) - 13.82*log10(hB) - a + (44.9-6.55*log10(hB))*log10(d) + C
    return L    

# 10*log10(x) = 10*lg(x)/lg(10) = lg(x)*10/lg(10)
# pow(10, x/10) = exp(lg(10)*x/10)=exp(x*lg(10)/10)

cdef double log10_10 = 0.23025850929940458 # log(10)/10.

cdef object cget_coverage(
    np.ndarray[DTYPE_t, ndim=2] S, 
    np.ndarray[DTYPE_t, ndim=2] B, 
    list[:] B_of_S,
    double RSRP_TH=-88,
    double SINR_TH=-3,
    double TERM_HEIGHT=1.5,
    double FREQ=900,
    double NOISE=-110):
    """
    Obtain the coverage number of sampled points x in S 
        that RSRP>=RSRP_TH and SINR>SINT_TH

    Parameter
    ========
    S: numpy ndarray[double, ndim=2, order/mode="c"], like [[x0 in km, y0 in km, e in m],...]
    B: BaseStation: numpy ndarray[double, ndim=2, order="c"], 
        like [[x0 in km, 
               y0 in km,
               e in m, 
               height in m, 
               azimuth in rad, 
               tilt in rad,
               power in dB ]]
    
    Return
    =======
    [covered_s_count, isCovered, bIndex, rsrp, sinr, noise]
    covered_s_count is the number of pointed covered. (scalar)
    The others are ndarray with the same length as S.
    isCovered: ndarray[bool], "True" means s in S is covered
    bIndex: the index of baseStation in B, with the most strong signal at the point s
    rstp: the signal strength
    sinr: the sinr
    noise: all the noise and signals from other base stations 
    """
    cdef:
        unsigned int num_samples = S.shape[0]
        unsigned int num_bases = B.shape[0]
    assert S.shape[1] == 3
    assert B.shape[1] == 7

    cdef:
        np.ndarray[np.uint8_t, ndim=1] isCovered = np.zeros(num_samples, dtype=np.uint8, order='C')
        np.ndarray[np.int64_t, ndim=1] bIndex = np.zeros(num_samples, dtype=np.int64, order='C')
        np.ndarray[DTYPE_t, ndim=1] rsrp = np.zeros(num_samples, dtype=DTYPE, order='C')
        np.ndarray[DTYPE_t, ndim=1] sinr = np.zeros(num_samples, dtype=DTYPE, order='C')
        np.ndarray[DTYPE_t, ndim=1] noise = np.zeros(num_samples, dtype=DTYPE, order='C')
    
    bIndex[:] = -1
    noise[:] = pow(10., NOISE/10.0)

    cdef:
        unsigned int covered_s_count = 0
        unsigned int si, bi
        DTYPE_t xs, ys, es, temp
        DTYPE_t xb, yb, eb, hb, hm, ab, tb, pb
        DTYPE_t d, h_diff, tilt_s, beta, azimuth_s, alpha, g, pathloss, v, pow_v

    params = []
    for si in range(num_samples):
        xs = S[si, 0]
        ys = S[si, 1]
        es = S[si, 2] 
        sblist = B_of_S[si]
        for bi in sblist:
        #for bi in range(num_bases):
            xb = B[bi,0]
            yb = B[bi,1]
            eb = B[bi,2]
            hb = B[bi,3]
            ab = B[bi,4]
            tb = B[bi,5]
            pb = B[bi,6]
            ## get rssi 
            d = sqrt((xb-xs)*(xb-xs)+(yb-ys)*(yb-ys)) # in km
            h_diff = hb + eb - ( TERM_HEIGHT + es ) # 高度差
            tilt_s = atan( h_diff / (d*1E3) ) 
            azimuth_s = atan2( xs-xb, ys-yb)

            beta = tb - tilt_s
            alpha =  azimuth_s - ab
            g = cgain(alpha, beta)
            
            if d < 0.030:
                pathloss = 70
            elif d > 1.0:
                pathloss = 1000
            else:
                hm = TERM_HEIGHT + es - eb #接收机海拔高度+接收机天线高度（1.5m）-发射机海拔高度
                if hb < hm: #如果发射机高度hb比接收机hm高度矮，那么两者高度互换
                    temp = hb
                    hb = hm
                    hm = temp
                if hm>10 or hm<1:
                    hm = 1.5
                pathloss = loss(d, hb, hm, FREQ, C=3)
                
            v = pb + g - pathloss
            pow_v = exp(v * log10_10)       #pow(10.0, v/10.0)
            if pow_v > rsrp[si]: # b is better
                noise[si] += rsrp[si]
                rsrp[si] = pow_v
                bIndex[si] = bi  # TODO: 
            else:
                noise[si] += pow_v
            params.append(
                    dict(si=si, bi=bi, tilt_s=tilt_s, azimuth_s=azimuth_s, pathloss=pathloss)
                )

        # end of for-loop b
        rsrp[si] = log(rsrp[si]) / log10_10 # 10.0 * log10( rsrp[si] ) 
        noise[si] = log(noise[si]) / log10_10
 
        sinr[si] = rsrp[si] - noise[si]
        isCovered[si] = sinr[si]>=SINR_TH and rsrp[si] >= RSRP_TH
        if isCovered[si]:
            covered_s_count += 1
    #end of for-loop s
    source_code = jinja2.Environment(
        loader=jinja2.FileSystemLoader('./')
        ).get_template("rsrpsinr_template.pyx").render(all_b_to_s_params=params)
    with codecs.open("rsrpsinrfix.pyx", "w", "utf-8") as f:
        f.write(source_code)
    return [covered_s_count, isCovered, bIndex, rsrp, sinr, noise] 


"""
Interface
"""
def gain(double alpha, double beta):
    return cgain(alpha, beta)
       
def get_coverage(
    np.ndarray[DTYPE_t, ndim=2] S, 
    np.ndarray[DTYPE_t, ndim=2] B, 
    object B_of_S,
    double RSRP_TH=-88,
    double SINR_TH=-3,
    double TERM_HEIGHT=1.5,
    double FREQ=900,
    double NOISE=-110):
    return cget_coverage(S, B, B_of_S,
        RSRP_TH, SINR_TH, TERM_HEIGHT, FREQ, NOISE)

#in-place

