#!/usr/bin/python
# -*- coding: utf-8 -*-


import gtk, gobject

import numpy as np

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, int, linspace, arange,sum,\
                  array, zeros, mean, diff, column_stack,\
                  unique, dot, logical_not, ones, concatenate
from numpy.random import normal, randint, random, shuffle

from scipy.spatial import cKDTree

import cairo, Image
from collections import defaultdict
from itertools import count


BACK = [0.1]*3
FRONT = [0.8,0.8,0.8,0.9]

CONTRASTA = [0.84,0.37,0] # orange
CONTRASTB = [0.53,0.53,1] # lightblue
CONTRASTC = [0.84,1,0]

PI = pi
TWOPI = 2.*pi

NMAX = 2*1e8
SIZE = 1000
ONE = 1./SIZE

#UPDATE_NUM = 1 # write image this often

STP = ONE
FARL  = 20*ONE # ignore nodes beyond this distance
NEARL = 3*ONE # do not attempt to approach neighbours close than this 

MID = 0.5
INIT_R = 0.0005
INIT_N = 100



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
    self.itt = 0


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

    if not self.steps%100:
      print self.steps

    return res



class Line(object):

  def __init__(self):

    self.X = zeros(NMAX,'float')
    self.Y = zeros(NMAX,'float')
    self.SV = {}
    self.SS = {}

    self.vnum = 0
    self.snum = 0
    self.sind = 0

  def _add_vertex(self,x,y):

    vnum = self.vnum
    self.X[vnum] = x
    self.Y[vnum] = y

    self.vnum += 1
    return self.vnum-1

  def update_tree(self):

    vnum = self.vnum

    xy = column_stack([self.X[:vnum],self.Y[:vnum]])
    self.tree = cKDTree(xy)

  def get_all_near_vertices(self,r):

    xy = column_stack([self.X[:self.vnum],self.Y[:self.vnum]])
    near_inds = self.tree.query_ball_point(xy,r)

    return near_inds

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

    self.SV[self.sind] = [a,b]
    self.sind += 1
    self.snum += 1
    return self.sind-1

  def _delete_segment(self,a):

    vv = self.SV[a]
    del(self.SV[a])
    self.snum -= 1

    connected_to = self.SS[a]
    for seg in connected_to:
      self.SS[seg] = [h for h in self.SS[seg] if not h==a]

    del(self.SS[a])

    return vv

  def split_segment(self,a):

    vv = self.SV[a]
    midx = (self.X[vv[1]] + self.X[vv[0]])*0.5
    midy = (self.Y[vv[1]] + self.Y[vv[0]])*0.5

    newv = self._add_vertex(midx,midy)

    connected_to_a = self.SS[a]
    connected_to_b = []
    connected_to_c = []

    for seg in connected_to_a:

      if vv[0] in self.SV[seg]:
        connected_to_b.append(seg)
      if vv[1] in self.SV[seg]:
        connected_to_c.append(seg)

    b = self._add_segment(vv[0],newv,connected_to=connected_to_b)
    connected_to_c.append(b)
    c = self._add_segment(vv[1],newv,connected_to=connected_to_c)

    self._delete_segment(a)

    return newv, [b,c]

  def get_tangent(self,a):
   
    try:

      vv = self.SV[a]
      sx = self.X[vv[1]] - self.X[vv[0]]
      sy = self.Y[vv[1]] - self.Y[vv[0]]
      dd = sqrt(sx*sx+sy*sy)
      scale = 1./dd

    except IndexError:
      return 0,0

    return sx*scale,sy*scale, dd



def init_circle(l,ix,iy,r,n):

  th = arange(n).astype('float')/float(n)*TWOPI
  xx = ix + cos(th)*r
  yy = iy + sin(th)*r

  vv = []
  for x,y in zip(xx,yy):
    vv.append(l._add_vertex(x,y))

  connected_to = []
  
  for i in xrange(len(vv)-1):
    
    seg = l._add_segment(vv[i],vv[i+1], connected_to=connected_to)
    if i == 0:
      first = seg

    connected_to = [seg]
  
  connected_to.append(first)
  l._add_segment(vv[0],vv[-1],connected_to=connected_to)


def growth(l):

  grow = []
  count = 0

  #near2 = l.get_all_near_vertices(FARL*2.)

  for s in l.SV.keys():

    ss = l.SS[s]
    
    try:
      t0x,t0y,dd0 = l.get_tangent(ss[0])
      t1x,t1y,dd1 = l.get_tangent(ss[1])
      dd = (dd0+dd1)*0.5

      dot = t0x*t1x+t0y*t1y
      kappa = 1.-np.abs(dot)
    except IndexError:
      kappa = 0.
   
    if random()<kappa**2 and dd>NEARL:
    #if dd>NEARL:
      grow.append(s)
      count += 1

  print 'total:',l.vnum,'new:',count

  new_vertices = []
  for g in grow:
    newv,_ = l.split_segment(g)
    new_vertices.append(newv)

  return new_vertices

def segment_attract(l,sx,sy):

  for s,vv in l.SV.iteritems():

    dx = l.X[vv[1]] - l.X[vv[0]]
    dy = l.Y[vv[1]] - l.Y[vv[0]]
    dd = sqrt(dx*dx+dy*dy)
    a = arctan2(dy,dx)

    if dd>NEARL:
      sx[vv[0]] += cos(a)
      sy[vv[0]] += sin(a)
      sy[vv[1]] -= sin(a)
      sx[vv[1]] -= cos(a)

def collision_reject(l,sx,sy):

  X = l.X
  Y = l.Y
  near = l.get_all_near_vertices(FARL)

  for k,(x,y,ii) in enumerate(zip(X,Y,near)):

    iii = set(ii)
    try:
      iii.remove(k)
    except KeyError:
      pass
    inds = list(iii)


    dx = x - X[inds]
    dy = y - Y[inds]
    dd = sqrt(square(dx)+square(dy))

    force = FARL-dd
    a = arctan2(dy,dx)

    sx[k] += sum(cos(a)*force)
    sy[k] += sum(sin(a)*force)



def main():

  L = Line()
  render = Render(SIZE)

  init_circle(L,MID,MID,INIT_R,INIT_N)

  SX = zeros(NMAX,'float')
  SY = zeros(NMAX,'float')


  def show(render,l):

    render.clear_canvas()
    render.ctx.set_source_rgba(*FRONT)
    render.ctx.set_line_width(ONE*2)
    for s,vv in l.SV.iteritems():
      render.line(l.X[vv[0]],l.Y[vv[0]],
                  l.X[vv[1]],l.Y[vv[1]])

    #render.ctx.set_source_rgba(*CONTRASTA)
    #vnum = l.vnum
    #render.circles(l.X[:vnum],l.Y[:vnum],
                   #ones(vnum,'float')*ONE*0.5)


  def step():

    L.update_tree()

    #if random()<0.1:

    new_vertices = growth(L)

    show(render,L)

    vnum = L.vnum
    SX[:vnum] = 0.
    SY[:vnum] = 0.

    segment_attract(L,SX[:L.vnum],SY[:vnum])
    collision_reject(L,SX[:L.vnum],SY[:vnum])

    SX[:vnum] *= STP
    SY[:vnum] *= STP

    L.X[:vnum] += SX[:vnum]
    L.Y[:vnum] += SY[:vnum]

    return True



  render.init_step(step)

  gtk.main()



if __name__ == '__main__':

  main()

