"""
    This file is used along with macad_gym to manage the Pylot docker container
"""
import argparse
import logging
import os
import signal
import subprocess
import time

import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        self.verbose = args.verbose

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
                    logger.info("get reset yes")
                    self.conn.set("reset", "no")
                    logger.info("set reset no")

                    # Kill previous pylot process
                    logger.info("kill pylot")
                    if pylot_process is not None and pylot_process.poll() is None:
                        pylot_process.terminate()

                    subprocess.run(
                        self.KILL_PYLOT,
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.conn.set("START_EGO", "0")
                    logger.info("pylot killed")

                    start_pylot_command = self.START_PYLOT_BASE_COMMAND.copy()
                    goal_loc = self.conn.get("pylot_end")
                    if goal_loc is not None:
                        logger.info("goal_loc: %s" % goal_loc)
                        start_pylot_command[-1] += f" --goal_location={goal_loc}"

                    ego_speed = self.conn.get("ego_speed")
                    if ego_speed is not None:
                        logger.info("ego_speed: %s" % ego_speed)
                        start_pylot_command[-1] += f" --target_speed={ego_speed}"

                    # Start new pylot process
                    time.sleep(1)
                    logger.info("start pylot")
                    pylot_process = subprocess.Popen(
                        start_pylot_command,
                        shell=False,
                        preexec_fn=os.setsid,
                        stdout=None if self.verbose else subprocess.DEVNULL,
                        stderr=None if self.verbose else subprocess.STDOUT,
                    )
                    logger.info("pylot started")

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
                logger.info("KeyboardInterrupt received, terminating the script...")
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pylot_manager = PylotManager(args)
    pylot_manager.run()
