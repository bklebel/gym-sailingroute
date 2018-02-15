import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import scipy.interpolate as interp
import time
import sys
import scipy


import boatfunc_disctrete as boatf

class SailingrouteEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, size=10):
    self.reset()
    self.size = size

    self.observation_space = spaces.Dict({"pos_start": spaces.Box(low=0, high=self.size, shape=(2,)), 
                                          "pos_goal":  spaces.Box(low=0, high=self.size, shape=(2,)), 
                                          "boat": spaces.Box(low=-40, high=40, shape=(18,20)),  
                                          "wind": spaces.Box(low=0, high=40, shape=(self.size,self.size)), # to be corrected for dict and stuff
                                          "depth": spaces.Box(low=0, high=30, shape=(self.size, self.size))}) 
    self.action_space = spaces.Discrete(360)

    self.state = None # observation 


  def step(self, action):
    pass 
  def reset(self):
    self.mesh, _, self.boat = boatf.generate_boat_complete()
    self.mesh_r, self.boat_r = boatf.boat_array_reduction(self.mesh, self.boat)

    self.wind = boatf.generate_wind_field(n_steps=self.size)

    self.state = {'pos_start' : boatf.generate_random_point(self.size), 
                  'pos_goal' : boatf.generate_random_point(self.size), 
                  'boat' : self.boat_r,
                  'wind' : self.wind
                  }
    

  def render(self, mode='human', close=False):
    pass
  def seed(s)
    np.random.seed(s)

  def speed(x, y, heading, weather, boat):
    return boatf.speed(x, y, heading, weather, boat)




class SailingrouteExtraHardEnv(SailingrouteEnv): 
  def __init__(size=500):
    super().__init__(size)

  def speed(x, y, heading, weather, boat):
    return boatf.speed(x, y, heading, weather, boat)*0.6





