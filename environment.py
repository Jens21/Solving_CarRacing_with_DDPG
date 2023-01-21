import gym
import matplotlib.pyplot as plt
import os
import numpy as np

class Environment():
    rewards = []
    reward_per_round = 0
    time_step = 0

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(12345)
        s = self.env.reset()
        self.s = self.edit_image(s)

    def edit_image(self, s):
        img_crop = s[:84]

        gray_img = np.zeros((84, 96))
        gray_img[img_crop[:, :, 0] == 255] = 1.
        gray_img[img_crop[:, :, 1] == 204] = 1.
        gray_img[(img_crop[:, :, 1] == 229) | (img_crop[:, :, 1] == 230)] = 1.

        return gray_img

    def step(self, action):
        s, r, done, _ = self.env.step(action)
        self.s = self.edit_image(s)
        self.reward_per_round += r
        self.time_step += 1

        if done or self.time_step>=600:
            self.time_step = 0
            self.rewards.append(self.reward_per_round)
            self.reward_per_round = 0

            s = self.env.reset()
            self.s = self.edit_image(s)

        speed = np.linalg.norm(self.env.car.hull.linearVelocity)

        return self.s, r, done, speed

    def plot_rewards(self, path):
        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(len(self.rewards)), self.rewards)
        plt.savefig(os.path.join(path, 'rewards.png'))
        plt.close()

    def close(self):
        self.env.close()

# if __name__ == '__main__':
#     env = Environment()
#     for _ in range(10):
#         s, r, done, speed = env.step([0, 1, 0])
#         plt.imsave('State.png', s, vmin=0, vmax=1)
#         print(speed)
#     env.plot_rewards('')
#     env.close()