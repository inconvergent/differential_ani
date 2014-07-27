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
def pyx_collision_reject(l,np.ndarray[double, mode="c",ndim=2] sx,double farl, int nz):

  cdef unsigned int vnum = l.vnum

  cdef np.ndarray[double, mode="c",ndim=2] X = l.X[:vnum,:]
  cdef np.ndarray[long, mode="c",ndim=1] VZ = l.VZ[:vnum]

  cdef dict neighbor_map = pyx_near_zone_inds(l.ZV,nz)

  cdef unsigned int v
  cdef unsigned int n
  cdef unsigned int num_neighbors
  cdef unsigned int neigh
  cdef list neighbors

  cdef double x
  cdef double y
  cdef double dx
  cdef double dy
  cdef double force
  cdef double nrm
  cdef double resx
  cdef double resy

  for v in range(vnum):
    resx = 0.
    resy = 0.
    x = X[v,0]
    y = X[v,1]
    neighbors = neighbor_map[VZ[v]]
    num_neighbors = len(neighbors)

    for n in range(num_neighbors):
      neigh = <unsigned int>neighbors[n]
      if neigh == v:
        continue
      dx = x-X[neigh,0]
      dy = y-X[neigh,1]
      nrm = norm(dx,dy)
      if nrm<=0 or nrm>farl:
        continue
      force = (farl-nrm)/nrm
      resx += dx*force
      resy += dy*force

    sx[v,0] += resx
    sx[v,1] += resy

  return


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pyx_segment_attract(l,np.ndarray[double, mode="c",ndim=2] sx,double near_limit):

  cdef unsigned int vnum = l.vnum
  cdef unsigned int sind = l.sind
  cdef unsigned int v1
  cdef unsigned int v2
  cdef unsigned int k

  cdef np.ndarray[double, mode="c",ndim=2] X = l.X[:vnum,:]
  cdef np.ndarray[long, mode="c",ndim=2] SV = l.SV[:sind,:]
  cdef np.ndarray[long, mode="c",ndim=1] SVMASK = l.SVMASK[:sind]

  cdef double x
  cdef double y
  cdef double dx
  cdef double dy
  cdef double dd
  cdef double forcex
  cdef double forcey

  for k in xrange(sind):

    if SVMASK[k]>0:

      v1 =  SV[k,0]
      v2 =  SV[k,1]

      dx = X[v2,0]-X[v1,0]
      dy = X[v2,1]-X[v1,1]
      dd = norm(dx,dy)
      
      if dd>near_limit:

        forcex = dx/dd
        forcey = dy/dd

        sx[v1,0] += forcex
        sx[v1,1] += forcey

        sx[v2,0] -= forcex
        sx[v2,1] -= forcey

  return

#@cython.wraparound(False)
#@cython.boundscheck(False)
#@cython.nonecheck(False)
#@cython.cdivision(True)
#def pyx_growth_branch(l,\
                      #np.ndarray[double, mode="c",ndim=1] rnd_split,\
                      #np.ndarray[double, mode="c",ndim=1] rnd_branch,\
                      #double near_limit):

  #cdef unsigned int sind = l.sind
  #cdef unsigned int vnum = l.vnum

  #cdef np.ndarray[long, mode="c",ndim=1] SVMASK = l.SVMASK[:sind]
  #cdef np.ndarray[long, mode="c",ndim=2] SV = l.SV[:sind,:]
  #cdef np.ndarray[double, mode="c",ndim=2] X = l.X[:vnum,:]

  #cdef unsigned int i
  #cdef unsigned int c
  #cdef unsigned int count_grow = 0
  #cdef unsigned int count_branch = 0

  #cdef double dx
  #cdef double dy
  #cdef double dd

  #cdef int *grow = <int *>malloc(sind*sizeof(int))
  #if not grow:
    #raise MemoryError()

  #cdef int *branch = <int *>malloc(sind*sizeof(int))
  #if not branch:
    #raise MemoryError()

  #try:
    #for i in range(sind):

      #if SVMASK[i]>0:

        #dx = X[SV[i,0],0] - X[SV[i,1],0]
        #dy = X[SV[i,0],1] - X[SV[i,1],1]
        #dd = norm(dx,dy)

        #if dd<=0.:
          #continue

        #if dd>near_limit*0.9:

          #if  rnd_split[i]<0.0001:
            #grow[count_grow] = i
            #count_grow += 1
            #continue

          #if rnd_branch[i]<0.0001:
            #branch[count_branch] = i
            #count_branch += 1
            #continue


    #for c in range(count_grow):
      #l.split_segment(grow[c])

    #for c in range(count_branch):
      #k = branch[c]
      #v1 = SV[k,0]
      #v2 = SV[k,1]

      #the = random()*3.14*2
      #x = X[v2,0] + np.sin(the)*near_limit
      #y = X[v2,1] + np.cos(the)*near_limit

      #l._add_vertex_segment([x,y],v2)

  #finally:
    #free(grow)
    #free(branch)

  #return


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pyx_growth(l,\
               np.ndarray[double, mode="c",ndim=1] rnd_split,\
               double near_limit):

  cdef unsigned int sind = l.sind
  cdef unsigned int vnum = l.vnum

  cdef np.ndarray[long, mode="c",ndim=1] SVMASK = l.SVMASK[:sind]
  cdef np.ndarray[long, mode="c",ndim=2] SV = l.SV[:sind,:]
  cdef np.ndarray[double, mode="c",ndim=2] X = l.X[:vnum,:]

  cdef unsigned int i
  cdef unsigned int c
  cdef unsigned int count_grow = 0

  cdef double dx
  cdef double dy
  cdef double dd

  cdef int *grow = <int *>malloc(sind*sizeof(int))
  if not grow:
    raise MemoryError()

  try:
    for i in range(sind):

      if SVMASK[i]>0:

        dx = X[SV[i,0],0] - X[SV[i,1],0]
        dy = X[SV[i,0],1] - X[SV[i,1],1]
        dd = norm(dx,dy)

        if dd<=0.:
          continue

        if dd>near_limit*0.9:

          if  rnd_split[i]<0.0001:
            grow[count_grow] = i
            count_grow += 1

    for c in range(count_grow):
      l.split_segment(grow[c])

  finally:
    free(grow)

  return


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline dict pyx_near_zone_inds(list zv,int nz):

  cdef dict neighbor_map = {}
  cdef unsigned int i
  cdef unsigned int j
  cdef unsigned int z


  cdef unsigned int i1
  cdef unsigned int i2
  cdef unsigned int i3
  cdef unsigned int i4
  cdef unsigned int i5
  cdef unsigned int i6
  cdef unsigned int i7
  cdef unsigned int i8
  cdef unsigned int i9

  cdef list inds 

  for i in xrange(nz):
    for j in xrange(nz):
      z = <unsigned int>((i+1)*(nz+2) + j+1)

      i1 = <unsigned int>(z-(nz+1))
      i2 = <unsigned int>(z-(nz+2))
      i3 = <unsigned int>(z-(nz+3))
      i4 = <unsigned int>(z-1)
      i5 = <unsigned int>(z+0)
      i6 = <unsigned int>(z+1)
      i7 = <unsigned int>(z+nz+1)
      i8 = <unsigned int>(z+nz+2)
      i9 = <unsigned int>(z+nz+3)

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
      neighbor_map[z] = inds
  
  return neighbor_map

