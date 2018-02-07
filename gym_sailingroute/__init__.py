from gym.envs.registration import register

register(
	id='sailingroute-v0', 
	entry_point='gym_sailingroute.envs:SailingrouteEnv', 
) 
register(
	id='sailingroute-extrahard-v0', 
	entry_point='gym_sailingroute.envs:SailingrouteExtraHardEnv', 
) 

