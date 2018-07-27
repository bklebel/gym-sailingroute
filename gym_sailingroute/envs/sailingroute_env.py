import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import math
import scipy.interpolate as interp
import time
import sys
import scipy

import os
import datetime

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt


PI = 3.14159265358979323846264338327

# import boatfunc_discrete as boatf



# class boatfunc():
#   def __init__(self):
#     # super(ClassName, self).__init__()
#     # self.arg = arg  
#     pass
def load_boat(file='bavaria_match.dat'): 
  """ loads a polar diagram with y axis being TWA, x axis being TWS
    returns a scipy function object, with following usage: 

    boat_speed = f(TWS, TWA)

    A TWA > 180 needs to be changed to 360-TWA. 

    the file needs to be like this (default bavaria match): 
    0     6 8  10   12  14  16    20  
    52     5.57 6.65 7.15 7.43 7.61 7.72 7.80 
    60     5.97 6.93 7.45 7.73 7.87 7.95 8.03 
    75     6.34 7.15 7.71 8.01 8.17 8.27 8.41 
    90     6.36 7.31 7.88 8.08 8.25 8.51 8.79 
    110    6.23 7.17 7.80 8.16 8.52 8.77 9.10 
    120    5.84 6.93 7.60 8.03 8.41 8.85 9.44 
    135    5.03 6.30 7.10 7.69 8.07 8.44 9.33 
    150    4.19 5.32 6.35 7.07 7.63 8.02 8.76 

  """
  bavaria = np.loadtxt(file)
  boat_data = bavaria[1:, 1:]
  x = bavaria[0]                 # tws
  y = np.transpose(bavaria)[0]   # twa
  a = np.zeros((1,boat_data.shape[1]))    # a, b, first and second are to interpolate to zero
  b = np.zeros((boat_data.shape[0]+1, 1))
  first = np.concatenate((a, boat_data))
  second = np.concatenate((b, first), axis=1)
  # print(second.shape, x.shape, y.shape)
  # return interp.interp2d(x, y, second, kind='cubic')
  mesh = np.meshgrid(x, y)
  bootf = interp.Rbf(mesh[0], mesh[1], second, function='cubic')
  return mesh, bootf, second


def cardioid_r(a, phi, factor, speed_max = 8):
  
  #print(factor)
  b = np.random.rand()*0.06
  c = np.random.rand()*0.06
  d = np.random.rand()*100
  e = np.random.rand()*100
  # b, c, d, e = 0.06, 0.06, 1, 1
  enhancing_f_1 = 5
  enhancing_f_2 = 3
  a = np.power(np.log10(a), 1.2)*2
  fun = 2*a*(1-np.cos(phi)+b*np.cos(phi*d/(2*PI))+c*np.sin(phi*e/(2*PI)))
  phi_90 = phi < np.radians(90);  phi_270 = phi > np.radians(270) ; 
  phi_mid1 = (phi > np.radians(135)) ; phi_mid2 = (phi < np.radians(225))
  # print(phi_mid, (phi > np.radians(135)).astype(int))
  fun = np.where(phi_90,  fun + enhancing_f_1*np.sin(2*phi), fun)
  fun = np.where(phi_270, fun - enhancing_f_1*np.sin(2*phi), fun)
  fun = np.where(phi_mid1 & phi_mid2, fun - enhancing_f_2*np.sin(2*(phi-3*PI/4)), fun)
  return fun*factor


  # def generate_boat(a, factor, cardioid=cardioid_r):
  #   phi = np.arange(0, 2*PI, PI/100)
  #   phi = np.radians(np.arange(0, 190, 10))
  #   return phi, cardioid(a, phi, factor)

def generate_boat_complete( cardioid=cardioid_r):
  wind_first  = np.linspace(1, 40, 40)
  wind_second = np.delete(np.roll(wind_first, 1), 0)
  wind_first  = np.delete(wind_first, 0)
  wind = np.random.uniform(wind_first, wind_second)
  phi = np.radians(np.linspace(1, 179, 179))
  boat = np.zeros((len(phi), len(wind), 1))
  factor = np.random.uniform(1, 3)
  for i in range(len(wind)):
    boat[:, i, 0] =  cardioid(phi=phi, a=wind[i], factor = factor)
  second = np.insert(np.insert(boat, 0, 0, axis=0), 0, 0, axis=1)
  mesh = np.meshgrid(np.insert(wind_first, 0, 0), np.insert(phi, 0, 0))
  # print(mesh[0].shape, second.shape)
  # print(type(mesh))
  return mesh, boat, second

def boat_array_reduction(mesh, boat,**kwargs):
  # boat = d['boat']
  # print(d['mesh'])
  # mesh = d['mesh']
  """this function takes in a boat array of shape (180, 40)
     and reduces it to (18, 20) - 180/10 and 40/2
     this is for the discrete version: these arrays are given to the NN, 
     while the original arrays are used for the underlying calculation of speed. 
     There, the respective wind will need to be rounded (down) to find the correct one
  """
  boat_new = boat[0::10, 0::2]
  mesh_new = [mesh[i][0::10, 0::2] for i in range(2)] 
  # print('reduction', boat_new.shape, mesh_new[0].shape, mesh_new[1].shape)
  return mesh_new, boat_new

def boat_to_array(boat, step_speed = 2, step_angle = 10, max_speed = 40, max_angle = 180, forplot=False):
  """returns a regular spaced array for boatspeed at TWS and TWA, 
     given the function boat(TWS, TWA) for a specific boat. 
  """
  TWS = np.arange(1, max_speed , step_speed)
  TWA = np.arange(1, max_angle , step_angle)
  # print(TWS, TWA, TWS.shape[0], TWA.shape[0])
  boat_array = np.zeros((int(TWA.shape[0]), int(TWS.shape[0])))
  # print(TWA, TWS)
  # print(boat_array.shape)
  for j in range(boat_array.shape[0]): 
    for i in range(boat_array.shape[1]): 
      # print(i, j)
      boat_array[j,i] = boat(TWS[i], TWA[j])
  if forplot: return boat_array
  return (TWA, TWS, boat_array)

def generate_random_sign(length):
  r = []
  for i in range(length): 
    # r1 = np.random.rand()
    a = 1 if np.random.rand() > 0.5 else -1
    r.append(a)
  return r

def generate_random_point(maximum, pos=False):
  x = np.random.rand()*(maximum-maximum/10) # *random[0]
  y = np.random.rand()*(maximum-maximum/10) # *random[1]
  if pos: 
      return np.array([[np.ceil(x)], [np.ceil(y)]])
  return [x, y]

def generate_obstacle_function(maximum, x, y):
  x_obstacle, y_obstacle = generate_random_point(maximum)
  alpha_obstacle, a_obstacle = np.random.rand()*2, 4e3

  p = -alpha_obstacle * np.power(1.5,-((x - x_obstacle)**2 / a_obstacle
                           + (y - y_obstacle)**2 / a_obstacle))
  return p

def generate_wind_field( maximum = 100, n_steps=10, plotting_scale = 1):
  '''generates a random wind field. first, random "obstacles" are introduced, 
     and a vector field is generated between them. 
     Then, every single wind component is turned by 15 degrees 
     to the right, approximately corresponding to the influence of 
     coriolis force on the northern hemisphere. 
     This is done by converting to speed and heading, adjusting the heading, 
     converting back to x and y components of the wind. 

     the "real size", computationally

  '''

  maxi_c  = np.complex(0, n_steps)
  x, y = np.mgrid[0:maximum:maxi_c, 0:maximum:maxi_c]

  a = generate_obstacle_function(maximum, x, y)
  b = generate_obstacle_function(maximum, x, y)
  # c = generate_obstacle_function(maximum, x, y)
  p = a - b # + c

  dx, dy = np.gradient(p) #, np.diff(y[:2, 0]), np.diff(x[0, :2]))

  tws = np.sqrt(dx**2+dy**2)
  twh = np.arctan2(dy,dx)
  for i in range(twh.shape[0]): 
    for j in range(twh.shape[1]): 
      twh[i,j] += 15

  dx = tws*np.cos(twh)*1.4e2
  dy = tws*np.sin(twh)*1.4e2

  u = interp.RectBivariateSpline(x[:,0], y[0,:], dx)
  v = interp.RectBivariateSpline(x[:,0], y[0,:], dy)

  # print(np.max(dx))
  skip = (slice(None, None, plotting_scale), slice(None, None, plotting_scale))

  return dict(u = u, dx=dx, v = v, dy=dy, x=x, y=y, tws=tws, twh=twh, skip=skip)

def deghead2rad(heading):
  '''sailing angle to math angle conversion
     math angle in radians       
  '''
  return np.radians(90-heading)
  
def speed_continuos(x, y, heading, weather, boat):
  """ Calculates the boat speed at a given time, 
      given the complete weather data (predicted at a specific time, for 240h in the future), polardiagram and heading
      boat takes a function! 

      tws = sqrt(u2+v2)
      twh = arctan2(v/u).
      twa = abs(heading - twh) 
  """
  u = weather['u'](x, y)[0][0] # still broken !!!!!!
  v = weather['v'](x, y)[0][0]

  tws = np.sqrt(u**2+v**2)
  twh = -PI/2-np.arctan2(v,u)
  twa = abs(heading - np.degrees(twh) )
  if twa > 180: 
    twa = abs(360-twa)
  assert twa < 180

  return boat(tws, twa), twa, tws, np.degrees(twh)

def _speed_slow(x, y, heading, weather, boat):
  """ Calculates the boat speed at a given time, 
      given the complete weather data (predicted at a specific time, for 240h in the future), polardiagram and heading
      boat takes an array! 

      tws = sqrt(u2+v2)
      twh = arctan2(v/u).
      twa = abs(heading - twh) 

      does it in a discrete fashion, rounding tws and twh, and searching for the corresponding boatspeed 
      in the discrete boat-array
  """
  u = weather['u'](x, y)[0][0] # still broken !!!!!! - really? 
  v = weather['v'](x, y)[0][0]

  tws = int(np.sqrt(u**2+v**2))
  twh = int(-PI/2-np.arctan2(v,u))
  twa = abs(heading - np.degrees(twh) )
  tws, twa = int(np.ceil(tws)), int(np.ceil(twa))
  twa = abs(740-twa) if twa > 360 else twa 
  twa = abs(359-twa) if twa > 179 else twa

  assert twa <= 179
  speedo = boat[twa, np.abs(boat[twa]-tws).argmin()][0]
  
  return speedo, twa, tws, np.degrees(twh) 
  # check this for correct tws/twa mapping to boat-array! 
  # print('twa',  np.degrees(twh), tws, twa, heading, speedo, x, y )

def _speed(x, y, heading, weather, boat):
  u = weather['u'](x, y)[0][0]  ;  v = weather['v'](x, y)[0][0]
  tws = int(np.sqrt(u**2+v**2))            ;  twh = int(-PI/2-np.arctan2(v,u))
  twa = abs(heading - np.degrees(twh) )    ;  tws, twa = int(np.ceil(tws)), int(np.ceil(twa))
  twa = abs(740-twa) if twa > 360 else twa ;  twa = abs(359-twa) if twa > 179 else twa
  assert twa <= 179
  speedo = boat[twa, np.abs(boat[twa]-tws).argmin()][0]
  return speedo, twa, tws, np.degrees(twh) 


def goal_heading(start, goal):
  vector = goal-start    ;   norm = math.sqrt(vector[0]**2+vector[1]**2)
  return math.acos(vector[1]/norm), norm

def VMG(goal_heading, heading, speed):
  vmg = speed*math.cos(np.radians(abs(goal_heading-heading)))
  # assert vmg > 0
  return vmg


def update_pos_slow( x, y, heading, speed):
  x += np.ceil(speed*math.cos(deghead2rad(heading)))
  y += np.ceil(speed*math.sin(deghead2rad(heading)))
  return [x,y]

def update_pos( x, y, heading, speed):
  return [x+np.ceil(speed*math.cos(deghead2rad(heading))), y+np.ceil(speed*math.sin(deghead2rad(heading)))]

class SailingrouteEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, size=200, timestep=1, resolution=20):
    self.size = size # this determines the physical size of the grid
    self.timestep = timestep # this determines "how fast time evolves" 
                             # 1 corresponds to 1h/step, 2 corresponds to 0.5h/step and so on
    self.resolution = resolution # this determines the resolution of the grid - which corresponds to the wind observation! 
    assert self.resolution == 20 # DO NOT CHANGE THIS! - 400 "pixel" for the NN (20x20)

    self.observation_space = spaces.Dict({"position": spaces.Box(low=0, high=self.size, shape=(2,2)), 
                                          "wind": spaces.Box(low=0, high=40, shape=(self.size,self.size)), # to be corrected for dict and stuff
                                          # "pos_goal":  spaces.Box(low=0, high=self.size, shape=(2,)), 
                                          #"heading_last" : spaces.Discrete() # finish up
                                          "boat": spaces.Box(low=-40, high=40, shape=(18,20,1))                                          
                                          #"depth": spaces.Box(low=0, high=30, shape=(self.size, self.size))
                                          }) 
    self.action_space = spaces.Discrete(360)
    self.reset()

    self.threshold = 20 # maximum distance to target to end the episode
    self.reward_range = (-1, 1)
    self.count = 0

    
    self.render_first = True
    self.ax = None
    
    self.printing = True

  def step(self, action):
    # _dist = self.state['position'][0] - self.state['position'][1]
    # start - goal
    step_punishment = 0.01
    # negative reward added at every non-successful step
    death_punishment = 0.5
    stuck_punishment = 0.4
    goal_reward = 5


    goal_head, goal_norm = goal_heading(self.state['position'][1], self.state['position'][0])
    # if np.sqrt(np.square(_dist[0]) + np.square(_dist[1]))  <= self.threshold: 
    if goal_norm <= self.threshold: 
      self._state['done'] = True
      self.course_traversed = []
      return self.state, goal_reward, True, {'goal': 'reached'}
    if self.state['position'][0][0] > self.size or self.state['position'][0][1] > self.size or \
       self.state['position'][0][0] < 0 or self.state['position'][0][1] < 0: 
      self._state['done'] = True
      self.course_traversed = []
      return self.state, -death_punishment, True, {'goal': 'missed'}

    speed = self.speed(self.state['position'][0][0], self.state['position'][0][1], 
                       self._state['wind'], self.boat, 
                       action)
    if speed <= 1: 
      self._state['zero_step_count'] += 1
      if self._state['zero_step_count'] > 5: 
        return self.state, -stuck_punishment, True, {'some': 'thing'}
    vmg_reward = VMG(goal_head, action, speed)/self.boat_max_speed*(step_punishment*4)-step_punishment*0.7
    # the norm was here chosen to be the boats maximum speed at any given wind, and any given wind angle. 
    # this could be changed to the maximum vmg possible at the current position - more calculations! 
    # calculate additional reward
    self.state['position'][0] =   update_pos(self.state['position'][0][0], self.state['position'][0][1], 
                                           action, speed)
    self.step_count += 1
    return self.state, -step_punishment+vmg_reward, False, {'some': 'thing'}
    # return observation, reward, done, info

  def reset(self):
    self.mesh, _, self.boat = generate_boat_complete()
    # self.mesh, _, self.boat = load_boat()
    dic = dict(mesh=self.mesh, boat=self.boat, test=0)
    self.mesh_r, self.boat_r = boat_array_reduction(**dic) #, self.boat)

    windfield = generate_wind_field(n_steps=self.resolution, maximum=self.size)
    self._state = {'wind' : windfield, 'done': False, 'zero_step_count': 0}
    self.state = {'wind' : np.array([windfield['twh'], windfield['tws']]),
                  'position' : np.array([generate_random_point(self.size, pos=True),generate_random_point(self.size, pos=True)]), 
                  'boat' : self.boat_r                  
                  }
    self.boat_max_speed = np.max(self.boat) # needs to be updated in case of function and continuity
    self.render_first = True
    self.course_traversed = []
    # self._state.update({'done': False})
    self.step_count = 0
    return self.state

  def seed(s):
    np.random.seed(s)

  def speed(self, x, y, weather, boat, heading):
    speed = _speed(x, y, heading, weather, boat)
    return speed[0]/self.timestep
    # TODO: reduce speed after turning corresponding to turn_angle/180
      # if turn_angle = 180, next speed step will be half as fast as without penalty,
      # if turn_angle = 0, no speed penalty is applied
      # speed = bf.speed()
      # speed -= speed*turn_angle/180
    # turn_angle = abs(heading - heading_last) # to be corrected
    # if turn_angle > 180: 
    #   turn_angle = abs(360-turn_angle)
    # speed -= speed*abs(turn_angle)/180*0.5
    # print(speed[0].shape)

  def _plotting(self, x, y, dx, dy, tws, skip, x_curr, y_curr, goal, first, axi, done, **kwargs):
    now = datetime.datetime.now()
    if first or done: 
      if plt: plt.close()
      fig, ax = plt.subplots()
      ax.quiver(x[skip], y[skip], dx[skip], dy[skip], tws[skip])
      ax.set(aspect=1, title='Course')
      plt.xlim(0, 200)
      plt.ylim(0, 200)
      goal = np.array([0,0])
      x_curr, y_curr = 0, 0
    else: 
      ax = axi
    ax.plot(x_curr, y_curr, 'b^--')
    ax.plot(goal[0], goal[1], 'r^')
    if self.printing: plt.savefig('pictures/pic_{}.png'.format(now.isoformat()))
    plt.draw()
    plt.pause(0.01)
    return ax

  def render(self, mode='human', close=False):
    self.render_first = True if self._state['done'] else self.render_first
    goal = self.state['position'][1]
    self.course_traversed.append([self.state['position'][0,0][0], self.state['position'][0,1][0]])
    # print(self.course_traversed[-1], goal)
    # print(self.ax, self.render_first, self._state['done'])
    self.ax = self._plotting(x_curr = np.array(self.course_traversed)[:,0], 
              y_curr = np.array(self.course_traversed)[:,1], 
              goal = goal, first = self.render_first, done = self._state['done'], 
              axi = self.ax, 
              **self._state['wind'])
    self.render_first = False

  def _plot_update(self):
    pass


  def _plot_boat(X, Y, boat):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    print(X.shape, Y.shape, boat.shape)
    # surf = ax.plot_surface(X, Y, boat, cmap = cm.coolwarm)
    surf = ax.plot_wireframe(X, Y, boat)
    # fig.colorbar(surf)
    plt.savefig('test.png')
    plt.show()
    # plt.draw()
    # plt.pause(0.01)

  def _plot_boat_polar(boat_func=None, boat_array=None, boat_fun=False, boat_arr = False):
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")

    if boat_fun: 
      boat = boat_func
      TWA, TWS, boating = boat_to_array(boat)
      # print(TWA.shape, TWS.shape, boating.shape, np.transpose(boating).shape)
      for i in range(len(TWS)): 
        # print(TWA,np.transpose(boating)[i] )
        ax.plot(np.radians(TWA), np.transpose(boating)[i]) 
      plt.savefig('polar_function.png')
      plt.show()
    if boat_arr: 
      boat = boat_array
      TWA, TWS, boating = boat_to_array(boat_func)
      for i in range(boat.shape[1]): 
        # print(TWA,np.transpose(boating)[i] )
        ax.plot(np.radians(TWA), np.transpose(boat)[i]) 
      plt.savefig('polar_array.png')
      plt.show()  


class SailingrouteExtraHardEnv(SailingrouteEnv): 
  def __init__(size=500):
    super().__init__(size)

  def speed(x, y, heading, weather, boat):
    return _speed(x, y, heading, weather, boat)[0]/self.timestep*0.6





