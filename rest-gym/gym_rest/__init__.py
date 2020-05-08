from gym.envs.registration import register
""" 
toggle between different envs of the folder rest-gym/gym-rest/envs by 
uncommenting and commenting the register function
"""

register(
    id='Restoration_Env-v0',
    entry_point='gym_rest.envs:RestEnv',
)

# here could be more envs for harder games of the same style

#register(
#        id='RestEnvContinuous-v0',
#        entry_point='gym_rest.envs:RestEnvContinuous',
#        )

#register(
#        id='Rest_Minimal-v0',
#        entry_point='gym_rest.envs:RestEnvMinimal',
#        )