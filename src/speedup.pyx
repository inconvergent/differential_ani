#!/usr/bin/python

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt
from libc.math cimport pow


DINT = np.int
ctypedef np.int_t DINT_t
DFLOAT = np.float
ctypedef np.float_t DFLOAT_t

@cython.cdivision(True)
cdef inline float get_force(float lim,float a, float b):
  cdef float dd = sqrt(pow(a,2)+pow(b,2))
  if dd <= 0.:
    return 0.
  else:
    return (lim-dd)/dd

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def pyx_collision_reject(l,np.ndarray[double, mode="c",ndim=2] sx,float farl):

  cdef unsigned int vnum = l.vnum

  cdef np.ndarray[double, mode="c",ndim=1] X = l.X[:vnum,0].ravel()
  cdef np.ndarray[double, mode="c",ndim=1] Y = l.X[:vnum,1].ravel()

  near = l.get_all_near_vertices(farl)

  cdef unsigned int k
  cdef unsigned int c
  cdef unsigned int j
  cdef unsigned int ind

  cdef float x
  cdef float y
  cdef float dx
  cdef float dy
  cdef float force
  cdef float resx
  cdef float resy

  for j in range(vnum):
    k = <unsigned int>len(near[j])
    resx = 0.
    resy = 0.
    x = X[j]
    y = Y[j]
    for c in range(k):
      ind = <unsigned int>near[j][c]
      if ind == k:
        continue
      dx = x-X[ind]
      dy = y-Y[ind]
      force = get_force(farl,dx,dy)
      resx += dx*force
      resy += dy*force

    sx[j,0] += resx
    sx[j,1] += resy

