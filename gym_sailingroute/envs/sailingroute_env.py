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
                                          "heading_last" : spaces.Discrete() # finish up
                                          "boat": spaces.Box(low=-40, high=40, shape=(18,20)),  
                                          "wind": spaces.Box(low=0, high=40, shape=(self.size,self.size)), # to be corrected for dict and stuff
                                          "depth": spaces.Box(low=0, high=30, shape=(self.size, self.size))}) 
    self.action_space = spaces.Discrete(360)
 
    self.threshold = 0.01 # maximum distance to target to end the episode
    reward_range = (-100, 100)

    action_space = None
    observation_space = None

  def step(self, action):
    pass 
    info = None

    if abs(self.state['pos_start'] - self.state['pos_goal']) <= self.threshold: 
      return self.state, 100, True, None

    speed = self.speed(self.state['pos_start'][0], self.state['pos_start'][1], 
                       self.state['wind'], self.boat, 
                       self.state['heading_last'], action)

    # calculate additional reward
    self.state['pos_start'] =   update_pos(self.state['pos_start'][0], self.state['pos_start'][1], 
                                           action, speed)
                       

    return self.state, reward, done, info 
    # return observation, reward, done, info


  def reset(self):
    self.mesh, _, self.boat = boatf.generate_boat_complete()
    self.mesh_r, self.boat_r = boatf.boat_array_reduction(self.mesh, self.boat)

    self.state = {'pos_start' : boatf.generate_random_point(self.size), 
                  'pos_goal' : boatf.generate_random_point(self.size), 
                  'heading_last' : 0, 
                  'boat' : self.boat_r,
                  'wind' : boatf.generate_wind_field(n_steps=self.size)
                  }

    self.boat_max_speed = np.max(self.boat)

    return self.state
    

  def render(self, mode='human', close=False):
    pass
  def seed(s)
    np.random.seed(s)

  def speed(x, y, weather, boat, heading_last, heading):
    # TODO: reduce speed after turning corresponding to turn_angle/180
      # if turn_angle = 180, next speed step will be half as fast as without penalty,
      # if turn_angle = 0, no speed penalty is applied
      # speed = bf.speed()
      # speed -= speed*turn_angle/180
    speed = boatf.speed(x, y, heading, weather, boat)
    turn_angle = abs(heading - heading_last) # to be corrected

    if turn_angle > 180: 
      turn_angle = abs(360-turn_angle)

    return speed -= speed*abs(turn_angle)/180*0.5




class SailingrouteExtraHardEnv(SailingrouteEnv): 
  def __init__(size=500):
    super().__init__(size)

  def speed(x, y, heading, weather, boat):
    return boatf.speed(x, y, heading, weather, boat)*0.6





