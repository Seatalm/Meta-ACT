import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import random
from argparse import ArgumentParser

from description.utils.generate_episode_instructions import (
    filter_instructions,
    load_task_instructions,
    replace_placeholders,
    replace_placeholders_unseen,
)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    os.makedirs(args["save_path"], exist_ok=True)
    clear_cache_freq = args["clear_cache_freq"]

    def write_seed_file():
        seed_path = os.path.join(args["save_path"], "seed.txt")
        tmp_path = seed_path + ".tmp"
        with open(tmp_path, "w") as file:
            for sed in seed_list:
                file.write("%s " % sed)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, seed_path)

    def exist_hdf5(idx):
        file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
        return os.path.exists(file_path)

    def exist_instruction(idx):
        file_path = os.path.join(args["save_path"], "instructions", f"episode{idx}.json")
        return os.path.exists(file_path)

    def write_episode_instruction(episode_idx, episode_info):
        episode_params = episode_info.get("info", {}) if isinstance(episode_info, dict) else {}
        if not episode_params:
            raise RuntimeError(f"Episode {episode_idx} has empty instruction parameters")

        task_data = load_task_instructions(args["task_name"])
        max_num = int(args.get("language_num", 1000000))

        seen_templates = filter_instructions(task_data.get("seen", []), episode_params)
        unseen_templates = filter_instructions(task_data.get("unseen", []), episode_params)
        if not seen_templates and not unseen_templates:
            raise RuntimeError(f"Episode {episode_idx}: no valid instructions found")

        seen = []
        while len(seen) < max_num and seen_templates:
            for instruction in seen_templates:
                if len(seen) >= max_num:
                    break
                seen.append(replace_placeholders(instruction, episode_params))
            random.shuffle(seen_templates)

        unseen = []
        while len(unseen) < max_num and unseen_templates:
            for instruction in unseen_templates:
                if len(unseen) >= max_num:
                    break
                unseen.append(replace_placeholders_unseen(instruction, episode_params))
            random.shuffle(unseen_templates)

        instruction_dir = os.path.join(args["save_path"], "instructions")
        os.makedirs(instruction_dir, exist_ok=True)
        instruction_path = os.path.join(instruction_dir, f"episode{episode_idx}.json")
        tmp_path = instruction_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump({"seen": seen, "unseen": unseen}, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, instruction_path)
        print(f"\033[92m[Instruction] Episode {episode_idx} saved.\033[0m")

    def collect_episode_data(episode_idx, seed):
        print(f"\033[34mTask name: {args['task_name']}\033[0m")
        data_args = args.copy()
        data_args["need_plan"] = False
        data_args["render_freq"] = 0
        data_args["save_data"] = True

        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **data_args)

        traj_data = TASK_ENV.load_tran_data(episode_idx)
        data_args["left_joint_path"] = traj_data["left_joint_path"]
        data_args["right_joint_path"] = traj_data["right_joint_path"]
        TASK_ENV.set_path_lst(data_args)

        info_file_path = os.path.join(args["save_path"], "scene_info.json")

        if not os.path.exists(info_file_path):
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump({}, file, ensure_ascii=False)

        with open(info_file_path, "r", encoding="utf-8") as file:
            info_db = json.load(file)

        info = TASK_ENV.play_once()
        info_db[f"episode_{episode_idx}"] = info

        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump(info_db, file, ensure_ascii=False, indent=4)

        TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
        TASK_ENV.merge_pkl_to_hdf5_video()
        TASK_ENV.remove_data_cache()
        assert TASK_ENV.check_success(), "Collect Error"
        write_episode_instruction(episode_idx, info)
        print(f"\033[92m[Saved] Episode {episode_idx} saved. (seed = {seed})\033[0m")

    # =========== Collect Seed and Data ===========
    if not args["use_seed"]:
        print("\033[93m" + "[Start One-by-One Seed and Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        # If interrupted after seed collection but before data conversion, fill missing data first.
        if args["collect_data"]:
            for episode_idx, seed in enumerate(seed_list):
                if not exist_hdf5(episode_idx):
                    collect_episode_data(episode_idx, seed)
                elif not exist_instruction(episode_idx):
                    with open(os.path.join(args["save_path"], "scene_info.json"), "r", encoding="utf-8") as file:
                        info_db = json.load(file)
                    write_episode_instruction(episode_idx, info_db[f"episode_{episode_idx}"])

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    episode_idx = suc_num
                    print(f"simulate data episode {episode_idx} success! (seed = {epid})")
                    TASK_ENV.save_traj_data(episode_idx)
                    TASK_ENV.close_env()

                    if args["render_freq"]:
                        TASK_ENV.viewer.close()

                    if args["collect_data"]:
                        collect_episode_data(episode_idx, epid)

                    seed_list.append(epid)
                    write_seed_file()
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1
                    TASK_ENV.close_env()

                    if args["render_freq"]:
                        TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

        print(f"\nComplete data generation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

        if args["collect_data"]:
            print("\033[93m" + "[Start Data Collection]" + "\033[0m")
            st_idx = 0
            while exist_hdf5(st_idx):
                st_idx += 1

            for episode_idx in range(st_idx, min(args["episode_num"], len(seed_list))):
                collect_episode_data(episode_idx, seed_list[episode_idx])


if __name__ == "__main__":
    # Skip render test in headless environment
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
