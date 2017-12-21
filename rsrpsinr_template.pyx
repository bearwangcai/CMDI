#!python
#cython: boundscheck=False, cdivision=True, wraparound=False
#coding=utf8 

import cython
cimport cython

import numpy as np
cimport numpy as np

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
    cdef double f  = fabs(alpha) / pi
    cdef double C1 = f * ( antenna_h_rad(pi) - antenna_v_rad(pi-beta) )
    cdef double C2 = (1-f) * ( antenna_h_rad(0) - antenna_v_rad(beta) )
    return antenna_h_rad(alpha) - C1 - C2

cdef double log10_10 = 0.23025850929940458 # log(10)/10.

cdef inline void b_to_s(
    np.ndarray[DTYPE_t, ndim=2] B,
    DTYPE_t[:] rsrp,
    DTYPE_t[:] noise,
    np.int32_t[:] bIndex,
    unsigned int si,
    unsigned int bi,
    double tilt_s,
    double azimuth_s, 
    double pathloss):
    """
    基础函数，处理一个基站到采样点的效果
    """
    cdef:
        double ab, tb, pb, beta, alpha, g, v, pow_v
    ab = B[bi,4]
    tb = B[bi,5]
    pb = B[bi,6]
    beta = tb - tilt_s
    alpha =  azimuth_s - ab
    g = cgain(alpha, beta)
    v = pb + g - pathloss
    pow_v = exp(v * log10_10)
    if pow_v > rsrp[si]: # b is better
        noise[si] += rsrp[si]
        rsrp[si] = pow_v
        bIndex[si] = bi  # TODO: 
    else:
        noise[si] += pow_v

cdef object cget_coverage(
    np.ndarray[DTYPE_t, ndim=2] S,
    np.ndarray[DTYPE_t, ndim=2] B,
    double RSRP_TH,
    double SINR_TH,
    double NOISE
):
    """
    Obtain the coverage number of sampled points x in S 
        that RSRP>=RSRP_TH and SINR>SINT_TH
    Parameter
    ========
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
        np.ndarray[np.int32_t, ndim=1] bIndex = np.zeros(num_samples, dtype=np.int32, order='C')
        np.ndarray[DTYPE_t, ndim=1] rsrp = np.zeros(num_samples, dtype=DTYPE, order='C')
        np.ndarray[DTYPE_t, ndim=1] sinr = np.zeros(num_samples, dtype=DTYPE, order='C')
        np.ndarray[DTYPE_t, ndim=1] noise = np.zeros(num_samples, dtype=DTYPE, order='C')
    
    bIndex[:] = -1
    noise[:] = pow(10., NOISE/10.0)

    cdef:
        unsigned int covered_s_count = 0
    {% for i in all_b_to_s_params %} 
    b_to_s(B, rsrp, noise, bIndex, {{ i.si }}, {{ i.bi }}, {{ i.tilt_s }}, {{ i.azimuth_s }}, {{ i.pathloss }})   {% endfor %}
    rsrp = np.log(rsrp)/log10_10
    noise = np.log(noise)/log10_10
    sinr = rsrp - noise
    isCovered = (sinr>=SINR_TH) and (rsrp >= RSRP_TH)
    covered_s_count = np.count_nozero(isCovered)
    return [covered_s_count, isCovered, bIndex, rsrp, sinr, noise] 

def get_coverage(
    np.ndarray[DTYPE_t, ndim=2] S, 
    np.ndarray[DTYPE_t, ndim=2] B, 
    double RSRP_TH=-88,
    double SINR_TH=-3,
    double NOISE=-110):
    return cget_coverage(S, B,
        RSRP_TH, SINR_TH, NOISE)
    #TODO: RSRP_TH, SINR_TH, NOISE, num_samples shoule be templated.



