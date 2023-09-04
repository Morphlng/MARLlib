# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import click
import os
import shutil
import subprocess

import ray


def do_link(file_path, force=False, local_path=None, packagent=None):
    file_path = os.path.abspath(os.path.join(
        os.path.dirname(packagent.__file__), file_path))

    if local_path is None:
        local_path = os.path.join("..", file_path)
    local_home = os.path.abspath(os.path.join(os.path.dirname(__file__), local_path))

    if not os.path.isfile(file_path) and not os.path.isdir(file_path):
        print(f"{file_path} does not exist. Continuing to link.")

    assert os.path.exists(local_home), local_home

    if not force and not click.confirm(
            f"This will replace:\n  {file_path}\nwith "
            f"a copy of:\n  {local_home}",
            default=True):
        return

    if os.name == "nt":
        # Delete the existing file or directory
        try:
            shutil.rmtree(file_path)
        except FileNotFoundError:
            pass
        except OSError:
            os.remove(file_path)

        # Copy the new file or directory
        if os.path.isdir(local_home):
            shutil.copytree(local_home, file_path)
        elif os.path.isfile(local_home):
            shutil.copy2(local_home, file_path)
        else:
            print(f"{local_home} is neither directory nor file. Copy failed.")
    else:
        sudo = []
        if not os.access(os.path.dirname(file_path), os.W_OK):
            print("You don't have write permission "
                  f"to {file_path}, using sudo:")
            sudo = ["sudo"]
        print(
            f"Creating symbolic link from \n {local_home} to \n {file_path}"
        )
        subprocess.check_call(sudo + ["rm", "-rf", file_path])
        subprocess.check_call(sudo + ["ln", "-s", local_home, file_path])


def patch(args):

    do_link("rllib/execution/replay_buffer.py", force=args.yes, local_path="./rllib/execution/replay_buffer.py",
            packagent=ray)
    do_link("rllib/execution/train_ops.py", force=args.yes,
            local_path="./rllib/execution/train_ops.py", packagent=ray)

    # models
    do_link("rllib/models/preprocessors.py", force=args.yes, local_path="./rllib/models/preprocessors.py",
            packagent=ray)

    # policy
    do_link("rllib/policy/rnn_sequencing.py", force=args.yes, local_path="./rllib/policy/rnn_sequencing.py",
            packagent=ray)
    do_link("rllib/policy/torch_policy.py", force=args.yes,
            local_path="./rllib/policy/torch_policy.py", packagent=ray)

    # utils
    do_link("rllib/utils/exploration/ornstein_uhlenbeck_noise.py", force=args.yes,
            local_path="./rllib/utils/exploration/ornstein_uhlenbeck_noise.py", packagent=ray)
    do_link("_private/resource_spec.py", force=args.yes,
            local_path="./_private/resource_spec.py", packagent=ray)

    if args.pommerman:
        import pommerman

        do_link('graphics.py', force=args.yes,
                local_path='pommerman/graphics.py', packagent=pommerman)

        do_link("__init__.py", force=args.yes,
                local_path='pommerman/__init__.py', packagent=pommerman)

        do_link("forward_model.py", force=args.yes, local_path="pommerman/forward_model.py",
                packagent=pommerman)

        do_link("envs/v0.py", force=args.yes,
                local_path="pommerman/v0.py", packagent=pommerman)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Setup dev.")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="RLlib_patch.")
    parser.add_argument(
        "--pommerman", "-p", action="store_true", help="pommerman.")
    args = parser.parse_args()

    patch(args)
    print("finish soft link")
