from pyvirtualdisplay import Display
import argparse
import numpy as np
import torch as th

np.random.seed(12345)
th.manual_seed(54321)

from replaybuffer import ReplayBuffer
from environment import Environment
from network import Network

BUFFER_SIZE = 100_000
BATCH_SIZE = 64
N_UPDATE_INTERVAL = 500
N_WARMUP = 1_000
N_TOTAL = 2_000_000
GAMMA = 0.95
N_HEURISTIC_USAGE = 200_000

def main():
    env = Environment()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    network = Network(N_UPDATE_INTERVAL, GAMMA, N_HEURISTIC_USAGE)

    s = env.s
    speed = 0
    for itt in range(N_TOTAL):
        action = network.get_action(s, speed, itt)
        next_s, r, done, next_speed = env.step(action)
        if not done:
            network.add_sample_to_buffer(s, speed, next_s, next_speed, r, action, replay_buffer)
        s = next_s
        speed = next_speed

        if itt >= N_WARMUP:
            network.train(replay_buffer, BATCH_SIZE)

        if itt % 1_000 == 0 and itt>0:
            network.plot_losses('')
            env.plot_rewards('')
            print('Itteration: {}\tLoss critic: {}\tLoss actor: {}'.format(itt, network.losses_critic[-1], network.losses_actor[-1]))

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--display', action="store_true", help='a flag indicating whether training runs in a virtual environment')

    args = parser.parse_args()

    if args.display:
        display = Display(visible=0, size=(800, 600))
        display.start()
        print('Display started')

    main()

    if args.display:
        display.stop()