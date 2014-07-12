#!/usr/bin/python

from numpy import cos, sin, pi, sqrt, sort,\
                  square, linspace, arange, logical_and,\
                  array, zeros, diff, column_stack, row_stack,\
                  unique, logical_not, ones, concatenate, reshape

from numpy import max as npmax
from numpy import min as npmin
from numpy import abs as npabs
from numpy import sum as npsum


def collision_reject(l,sx,farl):

  vnum = l.vnum
  X = reshape(l.X[:vnum,0],(vnum,1))
  Y = reshape(l.X[:vnum,1],(vnum,1))
  near = l.get_all_near_vertices(farl)
  nrows = len(near)
  n_near = [len(a) for a in near]
  ncols = npmax(n_near)

  ind = zeros((nrows,ncols),'int')
  xforce_mask = zeros((nrows,ncols),'bool')

  for j,(ii,lii) in enumerate(zip(near,n_near)):

    ind[j,:lii] = ii
    xforce_mask[j,lii:] = True

  dx = -X[ind,:].squeeze()
  dy = -Y[ind,:].squeeze()
  dx += X
  dy += Y

  dd = square(dx)+square(dy)
  sqrt(dd,dd)

  dx /= dd
  dy /= dd

  force = farl-dd

  xforce_mask[dd == 0.] = True
  dx[xforce_mask] = 0.
  dy[xforce_mask] = 0.

  force[xforce_mask] = 0.
  
  dx *= force
  dy *= force

  sx[:,0] += npsum(dx,axis=1)
  sx[:,1] += npsum(dy,axis=1)

