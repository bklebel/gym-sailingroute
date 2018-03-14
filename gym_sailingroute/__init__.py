from gym.envs.registration import register

register(
	id='sailingroute-v0', 
	entry_point='gym_sailingroute.envs:SailingrouteEnv-v0', 
    max_episode_steps=200,
    reward_threshold=25.0,
) 
register(
	id='sailingroute-extrahard-v0', 
	entry_point='gym_sailingroute.envs:SailingrouteExtraHardEnv-v0', 
    max_episode_steps=100,
    reward_threshold=25.0,
) 

