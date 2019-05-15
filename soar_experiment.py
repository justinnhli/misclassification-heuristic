import math
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from pathlib import Path
from random import Random

from colors import NearestCentroid, CENTROIDS
from soar_utils.soar_utils import create_agent, SoarEnvironment, cli

Color = namedtuple('Color', 'name, red, green, blue')

NUM_TRAIN_CENTROIDS = 5
NUM_TEST_CENTROIDS = 7

class FakeTableTop(SoarEnvironment):

    NUM_OBJECTS = 20

    SIZE_FILE = Path(__file__).parent.joinpath('soar/sizes')
    SHAPE_FILE = Path(__file__).parent.joinpath('soar/shapes')

    def __init__(self, agent, seed=8675309):
        super().__init__(agent)
        self.time = -1
        self.rng = Random(seed)
        self.history = []
        self.showing = False
        self.color_classifier = NearestCentroid(CENTROIDS[:NUM_TRAIN_CENTROIDS])
        with self.SIZE_FILE.open() as fd:
            self.sizes = [line.strip() for line in fd.readlines()]
        with self.SHAPE_FILE.open() as fd:
            self.shapes = [line.strip() for line in fd.readlines()]
        self.min_diameter = 100
        self.max_diameter = 0
        self.answer = None

    def initialize_io(self):
        # this function is called automatically by the super class
        self.next()
        self.show_object()

    def show_object(self):
        assert not self.showing
        input_link = self.agent.input_link
        for attribute, value in self.history[-1].items():
            self.add_wme(input_link, attribute, value)
        self.showing = True

    def _next_object(self):
        color = self._next_color()
        diameter = self._next_diameter()
        shape = self._next_shape()
        color_name = self._name_color(color)
        size_name = self._name_size(diameter)
        return {
            'time': self.time,
            'red': color[0],
            'green': color[1],
            'blue': color[2],
            'color': color_name,
            'diameter': diameter,
            'size': size_name,
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

    def _next_diameter(self):
        diameter = self.rng.gauss(50, 10)
        if diameter < self.min_diameter:
            self.min_diameter = diameter
        if diameter > self.max_diameter:
            self.max_diameter = diameter
        return diameter

    def _name_size(self, diameter):
        chunk = (self.max_diameter - self.min_diameter) / len(self.sizes)
        if chunk == 0:
            return self.sizes[len(self.sizes) // 2]
        index = int((diameter - self.min_diameter) // chunk)
        index = min(index, len(self.sizes) - 1)
        return self.sizes[index]

    def _next_shape(self):
        return self.rng.choice(self.shapes)

    def remove_object(self):
        assert self.showing
        input_link = self.agent.input_link
        for attribute, value in self.history[-1].items():
            self.del_wme(input_link, attribute, value)
        self.showing = False

    def next(self):
        self.time += 1
        self.history.append(self._next_object())

    def update_io(self):
        for command in self.parse_output_commands():
            if command.name == 'next':
                self.remove_object()
                if self.time < self.NUM_OBJECTS:
                    self.next()
                    self.show_object()
                else:
                    self.color_classifier = NearestCentroid(CENTROIDS[:NUM_TEST_CENTROIDS])
                    self.show_quiz()
                command.add_status('complete')
            elif command.name == 'check':
                if self.answer['time'] == command.arguments['time']:
                    command.add_status('correct')
                    # FIXME what to do if correct?
                else:
                    command.add_status('incorrect')
            elif command.name == 'give-up':
                pass # FIXME

    def show_quiz(self):
        input_link = self.agent.input_link
        # find object that has changed
        candidates = []
        for candidate in self.history:
            # calculate new labels
            new_size = self._name_size(candidate['diameter'])
            new_color = self._name_color((candidate['red'], candidate['green'], candidate['blue']))
            # check that an object is different
            is_candidate = (
                candidate['size'] != new_size
                or candidate['color'] != new_color
            )
            if not is_candidate:
                continue
            # check that no object has that description
            has_same = False
            for obj in self.history:
                if obj['size'] == new_size and obj['color'] == new_color:
                    has_same = True
                    break
            if not has_same:
                candidates.append(candidate)
        # create input
        self.add_wme(input_link, 'quiz', 'yes')
        if candidates:
            target = self.rng.choice(candidates)
        else:
            target = self.rng.choice(self.history)
        new_size = self._name_size(target['diameter'])
        new_color = self._name_color((target['red'], target['green'], target['blue']))
        print(f'clues: {new_size} {new_color} {target["shape"]}')
        print(f'target: {target["size"]} {target["color"]} {target["shape"]}')
        self.add_wme(input_link, 'size', new_size)
        self.add_wme(input_link, 'color', new_color)
        self.add_wme(input_link, 'shape', target['shape'])
        self.answer = target


def set_agent_commands(_, agent, args):
    agent.execute_command_line('smem --enable')
    agent.execute_command_line('smem --init')
    if args.memory == 'episodic':
        agent.execute_command_line('epmem --enable')
        agent.execute_command_line('epmem --init')

def init_agemt_smem(env, agent, _):
    # load memory with colors
    wmes = []
    for color in CENTROIDS[:NUM_TEST_CENTROIDS]:
        name = color.name.replace(' ', '_')
        wmes.append(' '.join([
            f'(<{name}>',
            '^type color',
            f'^r {color.r}',
            f'^g {color.g}',
            f'^b {color.b}',
            f'^name {name})',
        ]))
    distances = defaultdict(list)
    for i, color1 in enumerate(CENTROIDS[:NUM_TEST_CENTROIDS]):
        name1 = color1.name.replace(' ', '_')
        distances[name1].append((0, name1))
        for j, color2 in enumerate(CENTROIDS[i+1:NUM_TEST_CENTROIDS]):
            name2 = color2.name.replace(' ', '_')
            distance = math.sqrt(
                (color1.r - color2.r)**2
                + (color1.g - color2.g)**2
                + (color1.b - color2.b)**2
            )
            distances[name1].append((distance, name2))
            distances[name2].append((distance, name1))
            wmes.append(' '.join([
                f'(<{"_".join(sorted([name1, name2]))}>',
                f'^color <{name1}>',
                f'^color <{name2}>',
                f'^distance {distance})',
            ]))
    for color, pairs in distances.items():
        order = [color for _, color in sorted(pairs)]
        for prv, nxt in zip(order[:-1], order[1:]):
            wmes.append(f'(<{prv}> ^{color} <{nxt}>)')
    # load memory with sizes
    for i, size1 in enumerate(env.sizes):
        wmes.append(f'(<{size1}> ^type size ^name {size1})')
        for j, size2 in enumerate(env.sizes):
            wmes.append(' '.join([
                f'(<{"_".join(sorted([size1, size2]))}>',
                f'^size <{size1}>',
                f'^size <{size2}>',
                f'^distance {abs(i - j)})',
            ]))
        order = [
            size2 for _, size2 in
            sorted(
                enumerate(env.sizes),
                key=(lambda pair: abs(i - pair[0]))
            )
        ]
        for prv, nxt in zip(order[:-1], order[1:]):
            wmes.append(f'(<{prv}> ^{size1} <{nxt}>)')
    # actually create the LTMs
    agent.execute_command_line(f'smem --add {{ {" ".join(wmes)} }}')


def create_agent_params(env, agent, args):
    input_link = agent.input_link
    params_wme = env.add_wme(input_link, 'params')
    env.add_wme(params_wme.value, 'memory', args.memory)
    env.add_wme(params_wme.value, 'strategy', args.strategy)
    env.add_wme(params_wme.value, 'random-seed', args.random_seed)
    env.add_wme(params_wme.value, 'depth', args.depth)
    agent.execute_command_line('load file soar/agent.soar')


def run_experiment(args):
    with create_agent() as agent:
        env = FakeTableTop(agent, seed=int(args.random_seed))
        set_agent_commands(env, agent, args)
        init_agemt_smem(env, agent, args)
        create_agent_params(env, agent, args)
        cli(agent)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--memory', choices=['semantic', 'episodic'], default='epmem')
    arg_parser.add_argument('--strategy', choices=['simple', 'exhaustive', 'heuristic'], default='heuristic')
    arg_parser.add_argument('--random-seed', default=8675309)
    arg_parser.add_argument('--depth', type=int, default=1)
    args = arg_parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
