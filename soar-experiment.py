from argparse import ArgumentParser
from ast import literal_eval
from collections import namedtuple
from pathlib import Path
from random import Random

from colors import NearestCentroid, CENTROIDS
from soar_utils.soar_utils import create_agent, SoarEnvironment, cli

Color = namedtuple('Color', 'name, red, green, blue')

class FakeTableTop(SoarEnvironment):

    SIZE_FILE = Path(__file__).parent.joinpath('soar/sizes')
    SHAPE_FILE = Path(__file__).parent.joinpath('soar/shapes')

    def __init__(self, agent):
        super().__init__(agent)
        self.time = 0
        self.rng = Random(8675309)
        self.curr_obj = None
        self.color_classifier = NearestCentroid(CENTROIDS[:20])
        with self.SIZE_FILE.open() as fd:
            self.sizes = [line.strip() for line in fd.readlines()]
        with self.SHAPE_FILE.open() as fd:
            self.shapes = [line.strip() for line in fd.readlines()]
        self.min_size = 100
        self.max_size = 0

    def initialize_io(self):
        # this function is called automatically by the super class
        self.next()
        self.show_object()

    def show_object(self):
        input_link = self.agent.input_link
        for attribute, value in self.curr_obj.items():
            self.add_wme(input_link, attribute, value)

    def _next_object(self):
        color = self._next_color()
        size = self._next_size()
        shape = self._next_shape()
        color_name = self._name_color(color)
        size_name = self._name_size(size)
        return {
            'red': color[0],
            'green': color[1],
            'blue': color[2],
            'color_name': color_name,
            'size': size,
            'size_name': size_name,
            'shape': shape,
        }

    def _next_color(self):
        return (
            self.rng.randrange(256),
            self.rng.randrange(256),
            self.rng.randrange(256),
        )

    def _name_color(self, color):
        return self.color_classifier.classify_one(color).name

    def _next_size(self):
        size = self.rng.gauss(50, 10)
        if size < self.min_size:
            self.min_size = size
        if size > self.max_size:
            self.max_size = size
        return size

    def _name_size(self, size):
        chunk = (self.max_size - self.min_size) / len(self.sizes)
        if chunk == 0:
            return self.sizes[len(self.sizes) // 2]
        index = int((size - self.min_size) // chunk)
        return self.sizes[index]

    def _next_shape(self):
        return self.rng.choice(self.shapes)

    def remove_object(self):
        input_link = self.agent.input_link
        for attribute, value in self.curr_obj.items():
            self.del_wme(input_link, attribute, value)

    def next(self):
        self.time += 1
        self.curr_obj = self._next_object()

    def update_io(self):
        commands = self.parse_output_commands()
        for command in commands:
            if command.name == 'next':
                command.add_status('complete')
                self.remove_object()
                self.next()
                self.show_object()

def create_agent_params(env, agent, args):
    input_link = agent.input_link
    params_wme = env.add_wme(input_link, 'params')
    env.add_wme(params_wme.value, 'experiment', args.experiment)
    agent.execute_command_line('load file soar/agent.soar')
    if args.experiment == 'semantic':
        agent.execute_command_line('smem --enable')
        agent.execute_command_line('smem --init')
    else:
        agent.execute_command_line('epmem --enable')
        agent.execute_command_line('epmem --init')

def run_experiment(args):
    with create_agent() as agent:
        env = FakeTableTop(agent)
        create_agent_params(env, agent, args)
        cli(agent)

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('experiment', choices=['semantic', 'episodic', 'strategic'])
    args = arg_parser.parse_args()
    run_experiment(args)

if __name__ == '__main__':
    main()
