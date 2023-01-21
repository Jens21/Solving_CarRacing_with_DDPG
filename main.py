from pyvirtualdisplay import Display
import argparse

from replaybuffer import ReplayBuffer
from environment import Environment

BUFFER_SIZE = 10

def main():
    env = Environment()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--display', action="store_true", help='a flag indicating whether training runs in a virtual environment')

    args = parser.parse_args()

    if args.display:
        display = Display(visible=0, size=(800, 600))
        display.start()
        print('display started')

    main()

    if args.display:
        display.stop()