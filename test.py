import soccer_fours

env = soccer_fours.make(time_scale=10, render=True)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
team1_reward = 0
while True:
    obs, reward, done, info = env.step(
        {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
            3: env.action_space.sample(),
            4: env.action_space.sample(),
            5: env.action_space.sample(),
            6: env.action_space.sample(),
            7: env.action_space.sample(),
        }
    )

    team0_reward += reward[0] + reward[1] + reward[2] + reward[3]
    team1_reward += reward[4] + reward[5] + reward[6] + reward[7]
    if done["__all__"]:
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()