from pettingzoo.atari import entombed_competitive_v3

env = entombed_competitive_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    print(reward, termination, truncation, info)
    env.step(action)
    print(env.agent_iter())
env.close()