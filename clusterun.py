#!/usr/bin/env python3

import re
import sys
import subprocess
from ast import literal_eval
from datetime import datetime
from itertools import product
from textwrap import dedent


class PBSJob:

    # For more details, See the \`qsub\` manpage, or
    # http://docs.adaptivecomputing.com/torque/4-1-3/Content/topics/commands/qsub.htm
    TEMPLATE = dedent('''
        #!/bin/sh

        #PBS -N {name}
        #PBS -q {queue}
        #PBS -l {resources}
        #PBS -v {variables}
        #PBS -r n

        {commands}
    ''').strip()

    def __init__(self, name, commands, queue=None, venv=None):
        self.name = name
        self.commands = commands
        if queue is None:
            self.queue = 'justinli'
        else:
            self.queue = queue
        self.venv = venv
        # these are default for now; future changes may allow on-the-fly allocation
        self.resources = 'nodes=n006.cluster.com:ppn=1,mem=1000mb,file=4gb'

    def _generate_commands(self):
        commands = self.commands
        if self.venv is not None:
            prefixes = [
                'source "/home/justinnhli/.venv/{}/bin/activate"'.format(self.venv),
            ]
            suffixes = [
                'deactivate',
            ]
            commands = prefixes + commands + suffixes
        return '\n'.join(commands)

    def generate_script(self, variables=None):
        """Create a PBS job script.

        Arguments:
            variables (List[Tuple[str, obj]]): List of (key, value).

        Returns:
            str: The PBS job script.
        """
        if variables is None:
            variables = []
        return PBSJob.TEMPLATE.format(
            name=self.name,
            queue=self.queue,
            resources=self.resources,
            variables=','.join(f'{key}={value}' for key, value in variables),
            commands=self._generate_commands(),
        )

    def run(self, variables):
        """Create a single job.

        Arguments:
            variables (List[Tuple[str, obj]]): List of (key, value).
        """
        qsub_command = ['qsub', '-']
        subprocess.run(
            qsub_command,
            input=self.generate_script(*args, **kwargs).encode('utf-8'),
            shell=True,
        )

    def run_all(self, variable_space=None):
        """Create jobs for all variables.

        Arguments:
            variable_spaces (List[Tuple[str, List[obj]]]): List of (key, values).
        """
        if variable_space is None:
            variable_space = []
        keys = [pair[0] for key, values in variable_space]
        space = [pair[1] for key, values in variable_space]
        for values in product(*space):
            self.run(*zip(keys, values))


def run_cli(job_name, variables, commands, queue=None, venv=None):
    """Preview the job script and prompt for job start.

    Arguments:
        job_name (str): The name of the job.
        variables (List[Tuple[str, List[obj]]]): List of (key, values).
        commands (List[str]): Commands to run.
        queue (str): The queue to submit the jobs to.
        venv (str): The virtual environment to use.
    """
    pbs_job = PBSJob(job_name, commands, queue=queue, venv=venv)
    print(pbs_job.generate_script())
    print()
    print(40 * '-')
    print()
    # print variables
    space_size = 1
    warnings = []
    if variables:
        print('variables:')
        for var, vals in variables:
            print('    {}={}'.format(var, repr(vals)))
            for val in vals:
                if isinstance(val, str) and ',' in val:
                    warnings.append('variable "{}" has string value {} with a comma'.format(var, repr(val)))
            space_size *= len(vals)
    print('total invocations: {}'.format(space_size))
    if warnings:
        print()
        for warning in warnings:
            print('WARNING: ' + warning)
    print()
    print(40 * '-')
    print()
    # prompt confirmation
    try:
        response = input('Run jobs? (y/N) ')
    except KeyboardInterrupt:
        print()
        exit()
    if response.lower().startswith('y'):
        pbs_job.run_all(*variables)


def print_help():
    message = 'usage: {} [--<var>=<vals> ...] cmd [arg ...]'.format(
        sys.argv[0]
    )
    print(message)
    exit()


def parse_var(arg, force_list=True):
    var, vals = arg.split('=', maxsplit=1)
    var = var[2:].replace('-', '_')
    if not re.match('^[a-z]([_a-z0-9-]*?[a-z0-9])?$', var):
        raise ValueError('Invalid variable name: "{}"'.format(var))
    try:
        vals = literal_eval(vals)
    except ValueError:
        vals = vals
    if force_list and isinstance(vals, tuple([int, float, str])):
        vals = [vals]
    return var, vals


def parse_args():
    variables = []
    command = None
    kwargs = {}
    last_arg_index = 0
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg in ('-h', '--help'):
            print_help()
        elif arg.startswith('--'):
            if arg == '--':
                last_arg_index = i
                break
            var, vals = parse_var(arg)
            variables.append([var, vals])
        elif arg.startswith('-'):
            key, val = parse_var(arg, force_list=False)
            kwargs[key] = val
        else:
            break
        last_arg_index = i
    command = ' '.join(sys.argv[last_arg_index + 1:])
    return variables, command, kwargs


def main():
    variables, command, kwargs = parse_args()
    # print script
    job_name = 'from_cmd_' + datetime.now().strftime('%Y%m%d%H%M%S')
    commands = ['cd "$PBS_O_WORKDIR"', command]
    run_cli(job_name, variables, commands, **kwargs)


if __name__ == '__main__':
    main()
