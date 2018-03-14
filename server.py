"""
Server to communicate with RL part
"""
from __future__ import print_function, division, absolute_import

import os
import argparse

import zmq
from enum import Enum

from pipeline import stateRepresentationLearningCall, getBaseExpConfig, saveConfig, knnCall, getLogFolderName
from utils import createFolder


class Command(Enum):
    HELLO = 0
    LEARN = 1
    READY = 2
    ERROR = 3
    EXIT = 4


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Server to interface RL with state representation learning')
    parser.add_argument('--base_config', type=str, default="configs/default.json",
                        help='Path to overall config file, it contains variables independent from datasets (default: '
                             '/configs/default.json)')
    parser.add_argument('-p', '--port', type=int, default=7777)
    args = parser.parse_args()

    print('Starting up on port number {}'.format(args.port))
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    socket.bind("tcp://*:{}".format(args.port))

    print("Waiting for client...")
    path = os.path.abspath(__file__)
    socket.send_json({'command': Command.HELLO.value, 'path': path})
    msg = socket.recv_json()
    args.data_folder = msg.get('data_folder')
    print(args)
    print("Connected to client")

    exp_config = getBaseExpConfig(args)

    try:
        while True:
            print("Waiting for messages...")
            msg = socket.recv_json()

            try:
                # Convert to a command object
                command = Command(msg.get('command'))
            except ValueError:
                raise ValueError("Unknown command: {}".format(msg))

            if command == Command.LEARN:
                data_folder = msg.get('data_folder')

                createFolder("logs/{}".format(exp_config['data_folder']), "Dataset log folder already exist")

                exp_config['seed'] = msg.get('seed')
                # Update config
                exp_config['state_dim'] = msg.get('state_dim')
                log_folder, experiment_name = getLogFolderName(exp_config)
                exp_config['log_folder'] = log_folder
                exp_config['experiment_name'] = experiment_name
                # Save config in log folder
                saveConfig(exp_config, print_config=True)

                # Learn a state representation and plot it
                try:
                    ok = stateRepresentationLearningCall(exp_config)
                except RuntimeError:
                    ok = False
                if not ok:
                    socket.send_json({'command': Command.ERROR.value})
                    continue

                # Evaluate the representation with kNN
                knnCall(exp_config)

                path_to_model = os.path.abspath(exp_config['log_folder'])
                socket.send_json({'command': Command.READY.value, 'path': path_to_model})

            elif command == Command.EXIT:
                print("Exiting...")
                break
            else:
                raise NotImplementedError("Unsupported command: {}".format(command))
    except KeyboardInterrupt:
        pass

    socket.close()
