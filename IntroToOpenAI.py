import gym
import random
env_name = "CartPole-v1"
env = gym.make(env_name)

print("Observation space: ", env.observation_space)
print("Action Space : ", env.action_space)


class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size :", self.action_size)

    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle<0 else 1
        return action


if __name__ == '__main__':
    agent = Agent(env)
    state = env.reset()

    for _ in range(200):
        action  = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()