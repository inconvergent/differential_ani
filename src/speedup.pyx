#!/usr/bin/python

from __future__ import division

import numpy as np
from numpy.random import random
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport fabs

int = np.int
ctypedef np.int_t int_t
double = np.double
ctypedef np.double_t double_t

cdef inline double norm(double a, double b):
  cdef double dd = sqrt(pow(a,2)+pow(b,2))
  return dd

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pyx_collision_reject(l,np.ndarray[double, mode="c",ndim=2] sx,double farl):

  cdef unsigned int vnum = l.vnum

  cdef np.ndarray[double, mode="c",ndim=2] X = l.X[:vnum,:]

  near = l.get_all_near_vertices(farl)

  cdef unsigned int k
  cdef unsigned int c
  cdef unsigned int j
  cdef unsigned int ind

  cdef double x
  cdef double y
  cdef double dx
  cdef double dy
  cdef double force
  cdef double nrm
  cdef double resx
  cdef double resy

  for j in range(vnum):
    k = <unsigned int>len(near[j])
    resx = 0.
    resy = 0.
    x = X[j,0]
    y = X[j,1]
    nearj = near[j]
    for c in range(k):
      ind = <unsigned int>nearj[c]
      if ind == k:
        continue
      dx = x-X[ind,0]
      dy = y-X[ind,1]
      nrm = norm(dx,dy)
      if nrm<=0 or nrm>farl:
        continue
      force = (farl-nrm)/nrm
      resx += dx*force
      resy += dy*force

    sx[j,0] += resx
    sx[j,1] += resy

  return


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pyx_growth(l,np.ndarray[double, mode="c",ndim=1] rnd ,double near_limit):

  cdef unsigned int sind = l.sind
  cdef unsigned int vnum = l.vnum

  cdef np.ndarray[long, mode="c",ndim=1] SVMASK = l.SVMASK[:sind]
  cdef np.ndarray[long, mode="c",ndim=2] SV = l.SV[:sind,:]
  cdef dict SS = l.SS

  cdef np.ndarray[double, mode="c",ndim=2] X = l.X[:vnum,:]

  cdef unsigned int i
  cdef unsigned int s1
  cdef unsigned int s2

  cdef unsigned int c
  cdef unsigned int count = 0

  cdef double dx1
  cdef double dy1
  cdef double dx2
  cdef double dy2
  cdef double dd1
  cdef double dd2
  cdef double kappa2

  cdef int *grow = <int *>malloc(sind*sizeof(int))
  if not grow:
    raise MemoryError()

  try:
    for i in range(sind):

      if SVMASK[i]>0:

        s1 = <unsigned int>SS[i][0]
        s2 = <unsigned int>SS[i][1]
        
        dx1 = X[SV[s1,0],0]-X[SV[s1,1],0]
        dy1 = X[SV[s1,0],1]-X[SV[s1,1],1]
        dx2 = X[SV[s2,0],0]-X[SV[s2,1],0]
        dy2 = X[SV[s2,0],1]-X[SV[s2,1],1]

        dd1 = norm(dx1,dy1)
        dd2 = norm(dx2,dy2)

        if dd1<=0. or dd2<=0.:
          continue

        kappa2 = sqrt(1.-fabs( dx1/dd1*dx2/dd2+dy1/dd1*dy2/dd2 ))

        if (dd1+dd2)*0.5>near_limit and rnd[i]<kappa2:
          grow[count] = i
          count += 1

    for c in range(count):
      l.split_segment(grow[c])

  finally:
    free(grow)

  return



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def pyx_near_zone_inds(np.ndarray[long, mode="c",ndim=1] zz,zv,int nz):

  z_inds = []
  cdef int nz2 = nz+2
  cdef unsigned int z
  cdef unsigned int m

  cdef unsigned int i1
  cdef unsigned int i2
  cdef unsigned int i3
  cdef unsigned int i4
  cdef unsigned int i5
  cdef unsigned int i6
  cdef unsigned int i7
  cdef unsigned int i8
  cdef unsigned int i9

  m = zz.shape[0]

  cdef unsigned total = 0

  for i in xrange(m):
    z = zz[i]

    i1 = <unsigned int>(z-(nz2-1))
    i2 = <unsigned int>(z-nz2)
    i3 = <unsigned int>(z-(nz2+1))
    i4 = <unsigned int>(z-1)
    i5 = <unsigned int>(z+0)
    i6 = <unsigned int>(z+1)
    i7 = <unsigned int>(z+nz2-1)
    i8 = <unsigned int>(z+nz2)
    i9 = <unsigned int>(z+nz2+1)

    inds = []
    inds.extend(zv[i1])
    inds.extend(zv[i2])
    inds.extend(zv[i3])
    inds.extend(zv[i4])
    inds.extend(zv[i5])
    inds.extend(zv[i6])
    inds.extend(zv[i7])
    inds.extend(zv[i8])
    inds.extend(zv[i9])
    z_inds.append(inds)
    total += len(inds)
  
  return z_inds

