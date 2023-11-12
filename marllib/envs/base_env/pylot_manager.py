"""
    This file is used along with macad_gym to manage the Pylot docker container
"""
import argparse
import os
import subprocess
import signal
import time
import redis


class PylotManager:
    """
    This class is used to manage the Pylot docker container

    When macad env_config contains "use_redis": True, we will use this class to manage the Pylot docker container
    """

    def __init__(self, args):
        self.ue_host = args.ue_host
        self.ue_port = args.ue_port
        self.redis_host = args.redis_host
        self.redis_port = args.redis_port
        self.goal_location = "-2.128490,317.256927,0.556630"

        self.conn = redis.Redis(
            host=self.redis_host, port=self.redis_port, decode_responses=True
        )

        # Town01 map
        self.START_PYLOT_BASE_COMMAND = [
            "docker",
            "exec",
            "-it",
            "pylot",
            "bash",
            "-c",
            f"source /home/erdos/.bashrc;python3 pylot.py --flagfile=configs/scenarios/person_avoidance_frenet.conf --simulator_host={self.ue_host} --simulator_port={self.ue_port}",
        ]

        # The docker container of pylot should name "pylot"
        self.CHECK_PYLOT_ALIVE = [
            "docker",
            "exec",
            "-it",
            "pylot",
            "bash",
            "-c",
            "ps -ef | grep pylot.py | grep -v grep",
        ]
        self.KILL_PYLOT = [
            "docker",
            "exec",
            "-it",
            "pylot",
            "bash",
            "-c",
            "ps -ef | grep configs/ | awk '{print $2}' | xargs kill",
        ]
        self.TIMEOUT = 60

    def run(self):
        pylot_process = None
        while True:
            try:
                if self.conn.get("reset") == "yes":
                    print("get reset yes")
                    self.conn.set("reset", "no")
                    print("set reset no")

                    # Kill previous pylot process
                    print("kill pylot")
                    if pylot_process is not None and pylot_process.poll() is None:
                        pylot_process.terminate()

                    subprocess.run(
                        self.KILL_PYLOT,
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.conn.set("START_EGO", "0")
                    print("pylot killed")

                    start_pylot_command = self.START_PYLOT_BASE_COMMAND.copy()
                    goal_loc = self.conn.get("pylot_end")
                    if goal_loc is not None:
                        self.goal_location = goal_loc
                        start_pylot_command[-1] += f" --goal_location={self.goal_location}"

                    # Start new pylot process
                    time.sleep(1)
                    print("start pylot")
                    pylot_process = subprocess.Popen(
                        start_pylot_command, shell=False, preexec_fn=os.setsid
                    )
                    print("pylot started")

                if pylot_process is not None:
                    ret_code = pylot_process.poll()
                    if ret_code in [-2, 0, 130]:
                        raise KeyboardInterrupt
                    elif ret_code is not None:
                        raise RuntimeError(
                            "Pylot process exited with code {}".format(ret_code)
                        )

                time.sleep(0.1)
            except KeyboardInterrupt:
                print("KeyboardInterrupt received, terminating the script...")
                subprocess.run(
                    self.KILL_PYLOT,
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if pylot_process is not None and pylot_process.poll() is None:
                    os.killpg(os.getpgid(pylot_process.pid), signal.SIGTERM)
                    pylot_process.terminate()
                    pylot_process.wait(timeout=5)
                break
            except Exception:
                pylot_process = None

        self.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ue_host", type=str, default="172.17.0.1")
    parser.add_argument("--ue_port", type=int, default=2000)
    parser.add_argument("--redis_host", type=str, default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    args = parser.parse_args()

    pylot_manager = PylotManager(args)
    pylot_manager.run()
