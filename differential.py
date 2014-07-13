#!/usr/bin/python
# -*- coding: utf-8 -*-


import gtk, gobject

from numpy import cos, sin, pi, sqrt, sort,\
                  square, linspace, arange, logical_and,\
                  array, zeros, diff, column_stack, row_stack,\
                  unique, logical_not, ones, concatenate, reshape

from numpy import max as npmax
from numpy import min as npmin
from numpy import abs as npabs
from numpy import sum as npsum

from numpy.random import random, seed

from scipy.spatial import cKDTree

import cairo, Image
from collections import defaultdict
from itertools import count

from speedup.speedup import pyx_collision_reject
from speedup.speedup import pyx_growth

seed(3)

FNAME = './img/d_opt'


BACK = [0.1]*3
FRONT = [0.8,0.8,0.8,0.9]

CONTRASTA = [0.84,0.37,0] # orange
CONTRASTB = [0.53,0.53,1] # lightblue
CONTRASTC = [0.84,1,0]

PI = pi
TWOPI = 2.*pi

NMAX = 2*1e8
SIZE = 2000
ONE = 1./SIZE

STP = ONE*0.9
FARL  = 40.*ONE
NEARL = 3.*ONE

MID = 0.5
INIT_R = 0.0001
INIT_N = 100

RENDER_ITT = 1000 # redraw this often


class Render(object):

  def __init__(self,n):

    self.n = n

    self.__init_cairo()

    window = gtk.Window()
    window.resize(self.n, self.n)

    window.connect("destroy", self.__write_image_and_exit)
    darea = gtk.DrawingArea()
    darea.connect("expose-event", self.expose)
    window.add(darea)
    window.show_all()

    self.darea = darea

    self.num_img = 0


  def clear_canvas(self):

    self.ctx.set_source_rgb(*BACK)
    self.ctx.rectangle(0,0,1,1)
    self.ctx.fill()

  def __write_image_and_exit(self,*args):

    self.sur.write_to_png('on_exit.png')
    gtk.main_quit(*args)

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)
    ctx.set_source_rgb(*BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def init_step(self,e):

    self.step = e
    #gobject.timeout_add(5,self.step_wrap)
    gobject.idle_add(self.step_wrap)
    self.steps = 0

  def line(self,x1,y1,x2,y2):

    self.ctx.set_source_rgba(*FRONT)
    self.ctx.move_to(x1,y1)
    self.ctx.line_to(x2,y2)
    self.ctx.stroke()

  def circle(self,x,y,r,fill=False):

    self.ctx.arc(x,y,r,0,pi*2.)
    if fill:
      self.ctx.fill()
    else:
      self.ctx.stroke()

  def circles(self,xx,yy,rr,fill=False):

    if fill:
      action = self.ctx.fill
    else:
      action = self.ctx.stroke

    for x,y,r in zip(xx,yy,rr):
      self.ctx.arc(x,y,r,0,TWOPI)
      action()

  def expose(self,*args):

    cr = self.darea.window.cairo_create()
    cr.set_source_surface(self.sur,0,0)
    cr.paint()

  def step_wrap(self,*args):

    res = self.step()
    self.expose()
    self.steps += 1

    return res



class Line(object):

  def __init__(self):

    self.X = zeros((NMAX,2),'float')
    self.SV = zeros((NMAX,2),'int')
    self.SVMASK = zeros(NMAX,'int')
    self.SS = {}

    self.vnum = 0
    self.snum = 0
    self.sind = 0

  #@profile
  def _add_vertex(self,x):

    vnum = self.vnum
    self.X[vnum,:] = x

    self.vnum += 1
    return self.vnum-1
  
  #@profile
  def update_tree(self):

    vnum = self.vnum

    self.tree = cKDTree(self.X[:vnum,:])

  #@profile
  def get_all_near_vertices(self,r):

    near_inds = self.tree.query_ball_point(self.X[:self.vnum,:],r)

    return near_inds

  #@profile
  def _add_segment(self,a,b,connected_to=[]):

    for seg in connected_to:

      if self.SS.has_key(seg):
        self.SS[seg].append(self.sind)
      else:
        self.SS[seg] = [self.sind]

      if self.SS.has_key(self.sind):
        self.SS[self.sind].append(seg)
      else:
        self.SS[self.sind] = [seg]

    self.SV[self.sind,:] = [a,b]
    self.SVMASK[self.sind] = 1
    self.sind += 1
    self.snum += 1
    return self.sind-1

  #@profile
  def _delete_segment(self,a):

    vv = self.SV[a,:]
    self.SVMASK[a] = 0
    self.snum -= 1

    connected_to = self.SS[a]
    for seg in connected_to:
      self.SS[seg] = [h for h in self.SS[seg] if not h==a]

    del(self.SS[a])

    return vv

  #@profile
  def split_segment(self,a):

    vv = self.SV[a,:]
    midx = (self.X[vv[1],0] + self.X[vv[0],0])*0.5
    midy = (self.X[vv[1],1] + self.X[vv[0],1])*0.5
    #TODO: improve

    newv = self._add_vertex([midx,midy])

    connected_to_a = self.SS[a]
    connected_to_b = []
    connected_to_c = []

    for seg in connected_to_a:

      if vv[0] in self.SV[seg,:]:
        connected_to_b.append(seg)
      if vv[1] in self.SV[seg,:]:
        connected_to_c.append(seg)

    b = self._add_segment(vv[0],newv,connected_to=connected_to_b)
    connected_to_c.append(b)
    c = self._add_segment(vv[1],newv,connected_to=connected_to_c)

    self._delete_segment(a)

    return newv, [b,c]


def init_circle(l,ix,iy,r,n):

  th = sort(random(n)*TWOPI)
  xx = column_stack( (ix+cos(th)*r, iy+sin(th)*r) )

  vv = []
  for x in xx:
    vv.append(l._add_vertex(x))

  connected_to = []
  
  for i in xrange(len(vv)-1):
    
    seg = l._add_segment(vv[i],vv[i+1], connected_to=connected_to)
    if i == 0:
      first = seg

    connected_to = [seg]
  
  connected_to.append(first)
  l._add_segment(vv[0],vv[-1],connected_to=connected_to)

#@profile
def growth(l,near_limit):

  kvv,dx,dd = segment_lengths(l)
  alive_segments = l.SVMASK[:l.sind].nonzero()[0]
  
  k_mapping = {k:i for (i,k) in enumerate(alive_segments)}

  count = 0
  grow = []
  for s in alive_segments:

    ss = l.SS[s]

    s0 = k_mapping[ss[0]]
    s1 = k_mapping[ss[1]]

    length = (dd[s0]+dd[s1])*0.5
    dot = dx[s0,0]*dx[s1,0]+dx[s0,1]*dx[s1,1]
    kappa = 1.-npabs(dot)

    if random()<kappa**0.5 and length>near_limit:
      grow.append(s)
      count += 1
    
  new_vertices = []
  for g in grow:
    newv,_ = l.split_segment(g)
    new_vertices.append(newv)

  return new_vertices

#@profile
def segment_lengths(l):

  kvv = l.SV[:l.sind,:][l.SVMASK[:l.sind]>0,:]
  dx = diff(l.X[kvv,:],axis=1).squeeze()
  dd = square(dx)
  dd = npsum(dd,axis=1)
  sqrt(dd,dd)

  dx /= reshape(dd,(l.snum,1))
  
  return kvv,dx,dd


#@profile
def segment_attract(l,sx,nearl):

  kvv,dx,dd = segment_lengths(l)

  mask = dd>nearl
  sx[kvv[mask,0],:] += dx[mask,:]
  sx[kvv[mask,1],:] -= dx[mask,:]

#@profile
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


#@profile
def main():

  L = Line()
  render = Render(SIZE)

  init_circle(L,MID,MID,INIT_R,INIT_N)

  SX = zeros((NMAX,2),'float')


  #@profile
  def show(render,l):

    render.clear_canvas()
    render.ctx.set_source_rgba(*FRONT)
    render.ctx.set_line_width(ONE*2)
    for vv in l.SV[:l.sind,:][l.SVMASK[:l.sind]>0,:]:
      render.line(l.X[vv[0],0],l.X[vv[0],1],
                  l.X[vv[1],0],l.X[vv[1],1])


  #@profile
  def step():

    #new_vertices = growth(L,NEARL*1.1)
    new_vertices = pyx_growth(L,NEARL*1.1)

    L.update_tree()

    if not render.steps%RENDER_ITT:

      show(render,L)
      print 'steps:',render.steps,'vnum:',L.vnum,'snum:',L.snum
      
      fn = '{:s}_nearl{:0.0f}_itt{:07d}.png'.format(FNAME,FARL/ONE,render.steps)
      render.sur.write_to_png(fn)

    vnum = L.vnum
    SX[:vnum,:] = 0.

    segment_attract(L,SX[:L.vnum,:],NEARL)

    #collision_reject(L,SX[:L.vnum,:],FARL)
    pyx_collision_reject(L,SX[:L.vnum,:],FARL)

    SX[:vnum,:] *= STP
    L.X[:vnum,:] += SX[:vnum,:]

    return True



  render.init_step(step)

  gtk.main()


if __name__ == '__main__' :

  if True:

    import pstats, cProfile
    cProfile.run('main()','profile.profile')
    p = pstats.Stats('profile.profile')
    p.strip_dirs().sort_stats('cumulative').print_stats()

  else:

    main()

