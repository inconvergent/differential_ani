#!/usr/bin/python
# -*- coding: utf-8 -*-


import cairo, Image
import gtk, gobject

from numpy import cos, sin, pi, sqrt, sort, square,array, zeros, diff,\
                  column_stack,ones, reshape, linspace, arctan2

from numpy import sum as npsum
from numpy.random import random, seed

from itertools import count

from speedup.speedup import pyx_collision_reject
from speedup.speedup import pyx_growth_branch

seed(4)

FNAME = './img/xx'


BACK = [1]*3
FRONT = [0,0,0,0.7]

CONTRASTA = [0.84,0.37,0] # orange
CONTRASTB = [0.53,0.53,1] # lightblue
CONTRASTC = [0.84,1,0]

PI = pi
TWOPI = 2.*pi

NMAX = 2*1e8
SIZE = 1500
ONE = 1./SIZE

STP = ONE*0.9
FARL = 30.*ONE
NEARL = 3.*ONE
GROW_NEAR_LIMIT = 1.1*NEARL

MID = 0.5

LINEWIDTH = 3.*ONE


####


RENDER_ITT = 500 # redraw this often

ZONEWIDTH = FARL/ONE
ZONES = int(SIZE/ZONEWIDTH)



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

  def circle_stroke(self,x1,y1,x2,y2,r):

    dx = x1-x2
    dy = y1-y2
    dd = sqrt(dx*dx+dy*dy)

    n = int(dd/ONE)
    n = n if n>2 else 2

    a = arctan2(dy,dx)

    scale = linspace(0,dd,n)

    xp = x1-scale*cos(a)
    yp = y1-scale*sin(a)

    for x,y in zip(xp,yp):
      self.ctx.arc(x,y,r,0,pi*2.) 
      self.ctx.fill()

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
    self.VS = {}

    self.vnum = 0
    self.snum = 0
    self.sind = 0

    self.ZV = [[] for i in xrange((ZONES+2)**2)]
    self.VZ = zeros(NMAX,'int')

  def _add_vertex(self,x):
    """
    add vertex loctated at x.
    zone maps are updated.
    """

    vnum = self.vnum
    self.X[vnum,:] = x
    z = get_z(x,ZONES)
    self.ZV[z].append(vnum)
    self.VZ[vnum] = z

    self.vnum += 1
    return self.vnum-1

  def update_zone_maps(self):
    """
    check if vertices have changed zone, and update those that have.
    """
    
    vnum = self.vnum
    zz = get_zz(self.X[:vnum,:],ZONES)
    mask = (zz != self.VZ[:vnum]).nonzero()[0]

    for bad_v in mask:
      new_z = zz[bad_v]
      old_z = self.VZ[bad_v]

      new = [v for v in self.ZV[old_z] if v != bad_v]
      self.ZV[old_z] = new
      self.ZV[new_z].append(bad_v)

    self.VZ[mask] = zz[mask]

  def _add_segment(self,a,b):
    """
    add new segment between vertices a,b. 
    """

    self.SV[self.sind,:] = [a,b]
    self.SVMASK[self.sind] = 1

    add = make_dict_list_add(self.VS)

    add(a,self.sind)
    add(b,self.sind)

    self.sind += 1
    self.snum += 1

    return self.sind-1

  #def _add_vertex_segment(self,x,a):
    #"""
    #add new vertex x connected to vertex a with a new segment.
    #"""

    #v = self._add_vertex(x)
    #self._add_segment(a,b)

  def _delete_segment(self,a):
    """
    delete segment a and related connections.
    """

    vv = self.SV[a,:]
    self.SVMASK[a] = 0
    self.snum -= 1

    for v in vv:
      if self.VS.has_key(v):
        vs = [s for s in self.VS[v] if s!=a]
        if len(vs)>0:
          self.VS[v] = vs
        else:
          del(self.VS[v])

    return vv

  def split_segment(self,a):
    """
    add new vertex, v, at the middle of segment a with vertices: [v0,v1]
    creates new segments b,c such that: v0 -b- v -c- v1
    """

    vv = self.SV[a,:]
    midx = (self.X[vv[1],0] + self.X[vv[0],0])*0.5
    midy = (self.X[vv[1],1] + self.X[vv[0],1])*0.5
    #TODO: improve

    newv = self._add_vertex([midx,midy])
    self._delete_segment(a)
    b = self._add_segment(vv[0],newv)
    c = self._add_segment(vv[1],newv)

    return newv, [b,c]


def make_dict_list_add(d):

  def add(k,v):

    if d.has_key(k):
      d[k].append(v)
    else:
      d[k] = [v]

  return add

def get_z(x,nz):
  """
  find zone z of x. we have nz zones in each direction.
  """

  i = 1+int(x[0]*nz) 
  j = 1+int(x[1]*nz) 
  z = i*nz+j
  return z

def get_zz(xx,nz):
  """
  same as get_z for a vector of points.
  """

  ij = (xx*nz).astype('int')
  zz = ij[:,0]*(nz+2) + ij[:,1]+1
  return zz

def init_circle(l,ix,iy,r,n):

  th = sort(random(n)*TWOPI)
  rad = (0.9 + 0.1*(0.5-random(n)))*r
  xx = column_stack( (ix+cos(th)*rad, iy+sin(th)*rad) )

  vv = []
  for x in xx:
    vv.append(l._add_vertex(x))

  for i in xrange(len(vv)-1):
    seg = l._add_segment(vv[i],vv[i+1])

  l._add_segment(vv[0],vv[-1])


def init_horizontal_line(l,x1,x2,y1,y2,n):

  x = sort(x1+(x2-x1)*random(n))
  y = y1 + (y2-y1)*random(n)
  xx = column_stack((x,y))

  vv = []
  for x in xx:
    vv.append(l._add_vertex(x))
  
  for i in xrange(len(vv)-1):
    
    seg = l._add_segment(vv[i],vv[i+1])
    if i == 0:
      first = seg

def segment_lengths(l):

  kvv = l.SV[:l.sind,:][l.SVMASK[:l.sind]>0,:]
  dx = diff(l.X[kvv,:],axis=1).squeeze()
  dd = square(dx)
  dd = npsum(dd,axis=1)
  sqrt(dd,dd)

  dx /= reshape(dd,(l.snum,1))
  
  return kvv,dx,dd

def segment_attract(l,sx,nearl,scale=1.):

  kvv,dx,dd = segment_lengths(l)

  mask = dd>nearl
  sx[kvv[mask,0],:] += dx[mask,:]
  sx[kvv[mask,1],:] -= dx[mask,:]



def main():

  L = Line()
  render = Render(SIZE)

  #init_circle(L,MID,MID,0.001,50)
  init_horizontal_line(L,MID-0.2,MID+0.2,MID-0.001,MID+0.001,100)

  SX = zeros((NMAX,2),'float')

  def show(render,l):

    render.clear_canvas()
    render.ctx.set_source_rgba(*FRONT)
    render.ctx.set_line_width(LINEWIDTH)
    for vv in l.SV[:l.sind,:][l.SVMASK[:l.sind]>0,:]:
      render.circle_stroke(l.X[vv[0],0],l.X[vv[0],1],
                           l.X[vv[1],0],l.X[vv[1],1], ONE*2)


  def step():

    rnd1 = random(L.sind)
    rnd2 = random(L.sind)
    pyx_growth_branch(L,rnd1,rnd2,GROW_NEAR_LIMIT)

    L.update_zone_maps()

    if not render.steps%RENDER_ITT:

      show(render,L)
      print 'steps:',render.steps,'vnum:',L.vnum,'snum:',L.snum
     
      fn = '{:s}_nearl{:0.0f}_itt{:07d}.png'
      fn = fn.format(FNAME,FARL/ONE,render.steps)
      render.sur.write_to_png(fn)

    vnum = L.vnum

    SX[:vnum,:] = 0.
    segment_attract(L,SX[:vnum,:],NEARL)
    SX[:vnum,:] *= 0.5
    pyx_collision_reject(L,SX[:vnum,:],FARL,ZONES)

    SX[:vnum,:] *= STP
    L.X[:vnum,:] += SX[:vnum,:]

    return True



  render.init_step(step)

  gtk.main()


if __name__ == '__main__' :

  if False:

    import pstats, cProfile
    fn = './profile/profile'
    cProfile.run('main()',fn)
    p = pstats.Stats(fn)
    p.strip_dirs().sort_stats('cumulative').print_stats()

  else:

    main()

