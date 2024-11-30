import soccer_fours, soccer_threes, soccer_twos

num_per_team = 4
time_scale = 1
render = False

if num_per_team == 4:
    env = soccer_fours.make(time_scale=time_scale, render=render)
elif num_per_team == 3:
    env = soccer_threes.make(time_scale=time_scale, render=render)
elif num_per_team == 2:
    env = soccer_twos.make(time_scale=time_scale, render=render)
else:
    print(f"GOT UNEXPECTED NUMBER OF PLAYERS PER TEAM! {num_per_team}")
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)
team0_reward = 0
team1_reward = 0
env.reset()
while True:
    obs, reward, done, info = env.step(
        {
            i: env.action_space.sample() if i < 2 else [0, 0, 0] for i in range(2*num_per_team)
        }
    )
    for i in range(num_per_team):
        team0_reward += reward[i]
        team1_reward += reward[num_per_team+i]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()