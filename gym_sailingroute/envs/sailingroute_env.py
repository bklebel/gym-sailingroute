import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import scipy.interpolate as interp
import time
import sys
import scipy

import os

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
      return np.array([[x], [y]])
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

  return dict(u = u, dx=dx, v = v, dy=dy, x=x, y=y, tws=tws, skip=skip)

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

def _speed(x, y, heading, weather, boat):
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
  # print('twa',  np.degrees(twh), tws, twa, heading, speedo, x, y )
  return speedo, twa, tws, np.degrees(twh) 
  # check this for correct tws/twa mapping to boat-array! 

def update_pos( x, y, heading, speed):
  x += speed*np.cos(deghead2rad(heading))
  y += speed*np.sin(deghead2rad(heading))
  return [x,y]



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

    self.course_traversed = []
    self.render_first = True
    self.ax = None
    self._state = {'done': False}
    self.printing = True

  def step(self, action):
    _dist = self.state['position'][0] - self.state['position'][1]
    # start - goal

    if np.sqrt(np.square(_dist[0]) + np.square(_dist[1]))  <= self.threshold: 
      self._state['done'] = True
      self.course_traversed = []
      return self.state, 3, True, {'goal': 'reached'}
    if self.state['position'][0][0] > self.size or self.state['position'][0][1] > self.size or \
       self.state['position'][0][0] < 0 or self.state['position'][0][1] < 0: 
      self._state['done'] = True
      self.course_traversed = []
      return self.state, -3, True, {'goal': 'missed'}

    speed = self.speed(self.state['position'][0][0], self.state['position'][0][1], 
                       self._state['wind'], self.boat, 
                       action)
    # print(speed)
    # calculate additional reward
    self.state['position'][0] =   update_pos(self.state['position'][0][0], self.state['position'][0][1], 
                                           action, speed)
    # print('made step', self.count)
    self.count +=1
    return self.state, -0.01, False, {'some': 'thing'}
    # return observation, reward, done, info

  def reset(self):
    self.mesh, _, self.boat = generate_boat_complete()
    dic = dict(mesh=self.mesh, boat=self.boat, test=0)
    self.mesh_r, self.boat_r = boat_array_reduction(**dic) #, self.boat)

    windfield = generate_wind_field(n_steps=self.resolution, maximum=self.size)
    self._state = {'wind' : windfield, 'done': False}
    self.state = {'wind' : np.array([windfield['dx'], windfield['dy']]),
                  'position' : np.array([generate_random_point(self.size, pos=True),generate_random_point(self.size, pos=True)]), 
                  'boat' : self.boat_r                  
                  }
    self.boat_max_speed = np.max(self.boat) # needs to be updated in case of function and continuity
    return self.state

  def VMG(pos_curr, pos_goal, action, speed):
    pass

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

  def function():
    pass

  def _plotting(self, x, y, dx, dy, tws, skip, x_curr, y_curr, goal, first, axi, done, **kwargs):
    if done: plt.close()
    if first or done: 
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
    if self.printing: plt.savefig('pictures/pic_{}.png'.format(self.count))
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





