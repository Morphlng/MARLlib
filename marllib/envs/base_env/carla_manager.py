"""
    A Python script to manage the CARLA UE4 server. This should be used along with Macad-Gym to handle crashes.
"""

import argparse
import os
import subprocess
import time
import redis

IS_WINDOWS = os.name == "nt"


def kill_port(port):
    if IS_WINDOWS:
        find_port = "netstat -ano | findstr %s" % port
        result = subprocess.run(find_port, capture_output=True, text=True, shell=True)
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if line:
                pid = line.split()[-1]
                kill_pid(pid)
    else:
        find_port = ["lsof", "-i", ":%s" % port, "-t"]
        result = subprocess.run(find_port, capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                kill_pid(pid)


def kill_pid(pid):
    if IS_WINDOWS:
        find_kill = "taskkill -f -pid %s" % pid
    else:
        find_kill = ["kill", "-9", pid]

    print(find_kill)
    result = subprocess.run(find_kill, capture_output=True, text=True, shell=IS_WINDOWS)
    print("Killed process %s" % pid)


def main(args):
    while True:
        conn = redis.Redis(
            host=args.redis_host, port=args.redis_port, decode_responses=True
        )
        if conn.get("UE_START") == "1":
            kill_port(2000)
            time.sleep(1)
            conn.set("UE_START", 0)
            print(time.time())
            # Start Ue
            if IS_WINDOWS:
                proc = subprocess.Popen(args.ue_path, close_fds=True)
            else:
                proc = subprocess.Popen(["bash", args.ue_path], close_fds=True)
            print("UE started")
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_host", type=str, default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument(
        "--ue_path",
        type=str,
        default=os.environ.get("CARLA_SERVER", "~/software/CARLA_0.9.13/CarlaUE4.sh"),
    )
    args = parser.parse_args()

    main(args)
