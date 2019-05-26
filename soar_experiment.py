import math
from collections import namedtuple, defaultdict
from pathlib import Path
from random import Random

from permspace import PermutationSpace

from colors import NearestCentroid, CENTROIDS
from soar_utils.soar_utils import create_agent, SoarEnvironment, cli as agent_cli

Color = namedtuple('Color', 'name, red, green, blue')

class FakeTableTop(SoarEnvironment):

    SIZE_FILE = Path(__file__).parent.joinpath('soar/sizes')
    SHAPE_FILE = Path(__file__).parent.joinpath('soar/shapes')

    def __init__(self, agent, args):
        super().__init__(agent)
        # parameters
        self.num_objects = args.num_objects
        self.num_train_centroids = args.num_train_centroids
        self.num_test_centroids = args.num_test_centroids
        # variables
        self.time = -1
        self.rng = Random(args.random_seed)
        self.history = []
        self.showing = False
        self.color_classifier = NearestCentroid(CENTROIDS[:self.num_train_centroids])
        with self.SIZE_FILE.open() as fd:
            self.sizes = [line.strip() for line in fd.readlines()]
        with self.SHAPE_FILE.open() as fd:
            self.shapes = [line.strip() for line in fd.readlines()]
        self.min_diameter = 100
        self.max_diameter = 0
        self.answer = None
        # reports
        self.quiz_clues = None
        self.quiz_answer = None
        self.differences = 0
        self.num_checks = 0
        self.status = 'running'

    def initialize_io(self):
        # this function is called automatically by the super class
        self.next()
        self.show_object()
        self.add_wme(self.agent.input_link, 'stage', 'learn')

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
        return self.color_classifier.classify_one(color).name.replace(' ', '_')

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
                if self.time < self.num_objects:
                    self.next()
                    self.show_object()
                else:
                    self.color_classifier = NearestCentroid(CENTROIDS[:self.num_test_centroids])
                    self.show_quiz()
                command.add_status('complete')
            elif command.name == 'check':
                self.num_checks += 1
                if self.answer['time'] == command.arguments['time']:
                    command.add_status('correct')
                    self.status = 'correct'
                else:
                    command.add_status('incorrect')
            elif command.name == 'give-up':
                command.add_status('complete')
                self.status = 'gave-up'

    def show_quiz(self):
        input_link = self.agent.input_link
        self.del_wme(input_link, 'stage', 'learn')
        # find object that has changed
        candidates = []
        for candidate in self.history:
            # calculate new labels
            new_size = self._name_size(candidate['diameter'])
            new_color = self._name_color((candidate['red'], candidate['green'], candidate['blue']))
            # check that an object is different
            num_diff = 0
            if candidate['size'] != new_size:
                num_diff += 1
            if candidate['color'] != new_color:
                num_diff += 1
            if num_diff == 0:
                continue
            # check that no object has that description
            has_same = any(
                obj['size'] == new_size and obj['color'] == new_color
                for obj in self.history
            )
            if not has_same:
                candidates.append((num_diff, candidate))
        # create input
        self.add_wme(input_link, 'stage', 'quiz')
        if candidates:
            # FIXME prefer candidates that are different on two counts
            target = self.rng.choice(sorted(
                candidates,
                key=(lambda c: (c[0], c[1]['time'])),
                reverse=True,
            ))
            self.differences = target[0]
            target = target[1]
        else:
            target = self.rng.choice(sorted(
                self.history,
                key=(lambda c: c['time']),
            ))
        new_size = self._name_size(target['diameter'])
        new_color = self._name_color((target['red'], target['green'], target['blue']))
        self.quiz_clues = f'{new_size} {new_color} {target["shape"]}'
        self.quiz_answer = f'{target["size"]} {target["color"]} {target["shape"]}'
        '''
        for h in self.history:
            print(h['size'], h['color'], h['shape'])
        print()
        for h in candidates:
            print(h['size'], h['color'], h['shape'])
        print()
        print(f'clue: {self.quiz_clues}')
        print(f'answer: {self.quiz_answer} ({target["time"]})')
        '''
        self.add_wme(input_link, 'size', new_size)
        self.add_wme(input_link, 'color', new_color)
        self.add_wme(input_link, 'shape', target['shape'])
        self.answer = target


def set_agent_commands(_, agent, args):
    agent.execute_command_line('decide set-random-seed 8675309')
    agent.execute_command_line('smem --enable')
    agent.execute_command_line('smem --init')
    agent.execute_command_line('smem --set activation-mode base-level')
    agent.execute_command_line('smem --set spreading on')
    if args.memory == 'epmem':
        agent.execute_command_line('epmem --enable')
        agent.execute_command_line('epmem --init')


def init_agemt_smem(env, agent, args):
    # load memory with colors
    wmes = []
    for color in CENTROIDS[:args.num_test_centroids]:
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
    for i, color1 in enumerate(CENTROIDS[:args.num_test_centroids]):
        name1 = color1.name.replace(' ', '_')
        distances[name1].append((0, name1))
        for j, color2 in enumerate(CENTROIDS[i+1:args.num_test_centroids]):
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
    if hasattr(args, 'index_'): # HACK for PermSpace.Namespace
        keyvals = args._asdict().items()
    else:
        keyvals = vars(args).items()
    for key, val in keyvals:
        if key.endswith('_'):
            continue
        env.add_wme(params_wme.value, key.replace('_', '-'), val)
    agent.execute_command_line('load file soar/agent.soar')

ExpResult = namedtuple('ExpResult', 'clue answer differences status num_checks')

def run_experiment(args, interactive=False):
    with create_agent() as agent:
        env = FakeTableTop(agent, args)
        set_agent_commands(env, agent, args)
        init_agemt_smem(env, agent, args)
        create_agent_params(env, agent, args)
        if interactive:
            agent_cli(agent)
        else:
            #agent_cli(agent)
            agent.execute_command_line('run')
        return ExpResult(
            env.quiz_clues, env.quiz_answer,
            env.differences,
            env.status, env.num_checks,
        )

ExpParams = namedtuple('ExpParams', 'memory, strategy, random_seed, depth')

RNG = Random(8675309)

PSPACE = PermutationSpace(
    ['random_seed', 'memory', 'strategy', 'depth'],
    # environment
    num_objects=20,
    num_train_centroids=5,
    num_test_centroids=7,
    # agent
    memory=['smem', 'epmem'],
    strategy=['exact', 'exhaustive', 'heuristic'],
    random_seed=[RNG.random() for _ in range(100)],
    depth=[0, 100],
).filter(
    lambda strategy, depth:
        (strategy != 'heuristic' and depth == 0)
        or (strategy == 'heuristic' and depth != 0)
)

def cli():
    args = PSPACE.cli(defaults={
        # environment
        'num_objects': 20,
        'num_train_centroids': 5,
        'num_test_centroids': 7,
        # agent
        'memory': 'epmem',
        'strategy': 'heuristic',
        'random_seed': 8675309,
        'depth': 100,
    })
    print(args)
    print(run_experiment(args, interactive=True))


def main():
    for params in PSPACE:
        results = run_experiment(params)
        print('\t'.join(str(item) for item in [
            params.num_objects,
            params.num_train_centroids,
            params.num_test_centroids,
            params.memory,
            params.strategy,
            params.random_seed,
            params.depth,
            results.differences,
            results.status,
            results.num_checks,
        ]))

if __name__ == '__main__':
    #cli()
    main()
