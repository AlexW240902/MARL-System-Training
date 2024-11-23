from pettingzoo.atari import entombed_competitive_v3

env = entombed_competitive_v3.parallel_env(render_mode="human")
observations, infos = env.reset()
maxsteps = 1000

steps = 0

while env.agents and steps < maxsteps:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #print("Hey")
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards, terminations, truncations, infos)
    steps += 1
env.close()