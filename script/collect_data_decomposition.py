import sys
import os
import shutil
from datetime import datetime

# ================= 必须把路径设置放在最前面 =================
current_file_path = os.path.abspath(__file__)
root_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(root_directory)
sys.path.append("./")
# ==========================================================

from argparse import ArgumentParser
import time
import traceback
import json
import importlib
import yaml
import pdb
import pickle
from collections import OrderedDict
from sapien.render import clear_cache
import sapien.core as sapien
import random
import cv2
import h5py

from envs import *


sys.path.append("./")

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


def get_subtask_root(args, sub_id):
    return os.path.join(args["save_path"], f"data_subtask{sub_id}")


def get_subtask_data_dir(args, sub_id):
    return os.path.join(get_subtask_root(args, sub_id), "data")


def get_subtask_video_dir(args, sub_id):
    return os.path.join(get_subtask_root(args, sub_id), "video")


def get_subtask_traj_dir(args, sub_id):
    return os.path.join(get_subtask_root(args, sub_id), "_traj_data")


def get_subtask_instruction_dir(args, sub_id):
    return os.path.join(get_subtask_root(args, sub_id), "instructions")


def ensure_subtask_dirs(args, subtask_list):
    for sub_id in subtask_list:
        os.makedirs(get_subtask_data_dir(args, sub_id), exist_ok=True)
        os.makedirs(get_subtask_video_dir(args, sub_id), exist_ok=True)
        os.makedirs(get_subtask_traj_dir(args, sub_id), exist_ok=True)
        os.makedirs(get_subtask_instruction_dir(args, sub_id), exist_ok=True)


def get_traj_file_path(args, sub_id, traj_name):
    file_name = str(traj_name)
    if file_name.endswith(".pkl"):
        file_name = file_name[:-4]
    if not file_name.startswith("episode"):
        file_name = f"episode{file_name}"
    file_name = f"{file_name}.pkl"
    return os.path.join(get_subtask_traj_dir(args, sub_id), file_name)


def run_subtask_motion_plan(task_env, sub_id):
    motion_plan_func = getattr(task_env, f"_subtask{sub_id}_motion_plan")
    motion_plan_func()


def check_subtask_success(task_env, sub_id):
    check_success_func = getattr(task_env, f"_check_subtask{sub_id}_success")
    return bool(check_success_func())


def reset_after_subtask(task_env, sub_id):
    reset_func = getattr(task_env, "_reset_after_subtask", None)
    if reset_func is None:
        return True

    reset_success = reset_func(subtask_id=sub_id)
    return reset_success is not False


def update_subtask_state(task_env, sub_id, success):
    success = bool(success)

    if hasattr(task_env, "subtask_success"):
        task_env.subtask_success[sub_id - 1] = success
    if hasattr(task_env, "subtask_done"):
        task_env.subtask_done[sub_id - 1] = True


def get_subtask_list(task_env):
    has_subtask3 = (
        hasattr(task_env, "_subtask3_motion_plan")
        and hasattr(task_env, "_check_subtask3_success")
    )
    return [1, 2, 3] if has_subtask3 else [1, 2]



def save_subtask_traj_data(task_env, args, sub_id, traj_name):
    os.makedirs(get_subtask_traj_dir(args, sub_id), exist_ok=True)

    traj_data = {
        "left_joint_path": getattr(task_env, "left_joint_path", []),
        "right_joint_path": getattr(task_env, "right_joint_path", []),
    }

    traj_file_path = get_traj_file_path(args, sub_id, traj_name)
    with open(traj_file_path, "wb") as file:
        pickle.dump(traj_data, file)


def load_subtask_traj_data(args, sub_id, traj_name):
    traj_file_path = get_traj_file_path(args, sub_id, traj_name)
    with open(traj_file_path, "rb") as file:
        return pickle.load(file)


def save_subtask_data(task_env, args, sub_id):
    target_file_path = os.path.join(
        get_subtask_data_dir(args, sub_id),
        f"episode{task_env.ep_num}.hdf5",
    )
    target_video_path = os.path.join(
        get_subtask_video_dir(args, sub_id),
        f"episode{task_env.ep_num}.mp4",
    )
    merge_pkl_to_hdf5_video_custom(
        task_env, target_file_path, target_video_path)


def merge_pkl_to_hdf5_video_custom(task_env, target_file_path, target_video_path):
    cache_path = task_env.folder_path["cache"]
    scene_info = get_scene_info(task_env)

    from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video
    process_folder_to_hdf5_video(
        cache_path, target_file_path, target_video_path, scene_info
    )

    task_env.remove_data_cache()


def get_scene_info(task_env):
    subtask_done = getattr(task_env, "subtask_done", [])
    subtask_success = getattr(task_env, "subtask_success", [])

    scene_info = {
        "cluttered_table_info": getattr(task_env, "record_cluttered_objects", None),
        "texture_info": {
            "wall_texture": getattr(task_env, "wall_texture", None),
            "table_texture": getattr(task_env, "table_texture", None),
        },
        "table_z_bias": getattr(task_env, "table_z_bias", None),
        "table_xy_bias": task_env.table_xy_bias if hasattr(task_env, "table_xy_bias") else [0, 0],
        "random_background": getattr(task_env, "random_background", None),
        "cluttered_table": getattr(task_env, "cluttered_table", None),
        "random_light": getattr(task_env, "random_light", None),
        "random_head_camera_dis": getattr(task_env, "random_head_camera_dis", None),
        "random_table_height": getattr(task_env, "random_table_height", None),
        "current_subtask": getattr(task_env, "current_subtask", None),
        "subtask_done": subtask_done.copy() if hasattr(subtask_done, "copy") else subtask_done,
        "subtask_success": subtask_success.copy() if hasattr(subtask_success, "copy") else subtask_success,
    }

    if hasattr(task_env, "info") and isinstance(task_env.info, dict):
        scene_info.update(task_env.info)

    if hasattr(task_env, "get_asset_state"):
        asset_state = task_env.get_asset_state()
        if asset_state:
            scene_info["asset_state"] = asset_state

    return scene_info


def infer_instruction_params(task_env):
    """Recover instruction placeholders for decomposed motion plans.

    Some official tasks only populate self.info["info"] at the end of play_once().
    The decomposition collector calls subtask motion plans directly, so those
    fields can be empty even after a valid successful replay.
    """
    task_name = task_env.__class__.__name__

    if task_name == "place_can_basket_decomposition":
        return {
            "{A}": f"{task_env.can_name}/base{task_env.can_id}",
            "{B}": f"{task_env.basket_name}/base{task_env.basket_id}",
            "{a}": str(task_env.arm_tag),
        }

    if task_name == "place_burger_fries_decomposition":
        return {
            "{A}": f"006_hamburg/base{task_env.object1_id}",
            "{B}": f"008_tray/base{task_env.tray_id}",
            "{C}": f"005_french-fries/base{task_env.object2_id}",
        }

    if task_name == "put_object_cabinet_decomposition":
        return {
            "{A}": f"{task_env.selected_modelname}/base{task_env.selected_model_id}",
            "{B}": "036_cabinet/base0",
            "{a}": str(task_env.arm_tag),
            "{b}": str(task_env.arm_tag.opposite),
        }

    if task_name == "blocks_ranking_rgb_decomposition":
        return {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": "left",
            "{b}": "left",
            "{c}": "left",
        }

    if task_name == "place_dual_shoes_decomposition":
        return {
            "{A}": f"041_shoe/base{task_env.shoe_id}",
            "{B}": "007_shoe-box/base0",
        }

    if task_name == "stack_bowls_two_decomposition":
        return {
            "{A}": "002_bowl/base3",
            "{B}": "002_bowl/base3",
        }

    if task_name == "stack_blocks_two_decomposition":
        return {
            "{A}": "036_wooden_block/base0",
            "{B}": "036_wooden_block/base1",
        }

    if task_name == "lift_pot_decomposition":
        return {"{A}": f"{task_env.model_name}/base{task_env.model_id}"}

    if task_name == "dump_bin_bigbin_decomposition":
        return {"{A}": f"063_tabletrashbin/base{task_env.deskbin_id}"}

    return {}


def get_subtask_episode_params(episode_info, sub_id):
    subtask_info = episode_info.get("subtask_info", {})
    params = subtask_info.get(str(sub_id)) or subtask_info.get(sub_id)
    if not params:
        params = episode_info.get("info", {})
    return params or {}


def load_subtask_instruction_templates(args, sub_id):
    template_name = f"{args['task_name']}_subtask{sub_id}.json"
    template_path = os.path.join(
        root_directory, "description", "task_instruction", template_name
    )
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Missing subtask instruction template: {template_path}"
        )

    with open(template_path, "r", encoding="utf-8") as file:
        templates = json.load(file)
    for split in ("seen", "unseen"):
        if split not in templates or not isinstance(templates[split], list):
            raise ValueError(
                f"{template_path} must contain a list field named '{split}'"
            )
    return templates


def generate_subtask_instruction_payload(args, sub_id, episode_info):
    from description.utils.generate_episode_instructions import (
        extract_placeholders,
        filter_instructions,
        replace_placeholders,
        replace_placeholders_unseen,
    )

    episode_params = get_subtask_episode_params(episode_info, sub_id)
    if not episode_params:
        raise RuntimeError(
            f"Episode {episode_info.get('episode_idx', '?')} Subtask {sub_id} "
            "has empty instruction parameters"
        )

    templates = load_subtask_instruction_templates(args, sub_id)
    max_num = int(args.get("language_num", 1000000))
    used_placeholders = {
        placeholder
        for split in ("seen", "unseen")
        for instruction in templates[split]
        for placeholder in extract_placeholders(instruction)
    }
    if used_placeholders:
        episode_params = {
            key: value for key, value in episode_params.items()
            if key.strip("{}") in used_placeholders
        }

    filtered_seen = filter_instructions(templates["seen"], episode_params)
    filtered_unseen = filter_instructions(templates["unseen"], episode_params)
    if not filtered_seen and not filtered_unseen:
        raise RuntimeError(
            f"No valid instructions for task={args['task_name']} subtask={sub_id} "
            f"params={sorted(episode_params.keys())}"
        )

    seen = []
    while len(seen) < max_num and filtered_seen:
        for instruction in filtered_seen:
            if len(seen) >= max_num:
                break
            seen.append(replace_placeholders(instruction, episode_params))

    unseen = []
    while len(unseen) < max_num and filtered_unseen:
        for instruction in filtered_unseen:
            if len(unseen) >= max_num:
                break
            unseen.append(replace_placeholders_unseen(instruction, episode_params))

    unresolved = [
        text for text in seen + unseen
        if "{" in text or "}" in text
    ]
    if unresolved:
        raise RuntimeError(
            f"Unresolved instruction placeholders for subtask {sub_id}: "
            f"{unresolved[:3]}"
        )

    return {"seen": seen, "unseen": unseen}


def save_subtask_instruction(args, sub_id, episode_idx, episode_info):
    os.makedirs(get_subtask_instruction_dir(args, sub_id), exist_ok=True)
    instruction_path = os.path.join(
        get_subtask_instruction_dir(args, sub_id),
        f"episode{episode_idx}.json",
    )
    payload = generate_subtask_instruction_payload(args, sub_id, episode_info)
    atomic_write_json(instruction_path, payload)


def save_episode_instructions(args, subtask_list, episode_idx, episode_info):
    for sub_id in subtask_list:
        save_subtask_instruction(args, sub_id, episode_idx, episode_info)


def read_seed_list(save_path):
    seed_file_path = os.path.join(save_path, "seed.txt")
    with open(seed_file_path, "r", encoding="utf-8") as file:
        return [int(seed) for seed in file.read().split()]


def atomic_write_text(file_path, content):
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(content)
    os.replace(tmp_path, file_path)


def atomic_write_json(file_path, content):
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4, ensure_ascii=False)
    os.replace(tmp_path, file_path)


def get_episode_artifact_paths(save_path, subtask_list, episode_idx):
    artifacts = []
    for sub_id in subtask_list:
        artifacts.extend([
            os.path.join(
                save_path, f"data_subtask{sub_id}", "_traj_data", f"episode{episode_idx}.pkl"
            ),
            os.path.join(
                save_path, f"data_subtask{sub_id}", "data", f"episode{episode_idx}.hdf5"
            ),
            os.path.join(
                save_path, f"data_subtask{sub_id}", "video", f"episode{episode_idx}.mp4"
            ),
            os.path.join(
                save_path, f"data_subtask{sub_id}", "instructions", f"episode{episode_idx}.json"
            ),
        ])
    return artifacts


def clear_episode_artifacts(save_path, subtask_list, episode_idx):
    for file_path in get_episode_artifact_paths(save_path, subtask_list, episode_idx):
        if os.path.exists(file_path):
            os.remove(file_path)


def collect_numbered_episode_indices(dir_path, prefix, suffix):
    if not os.path.isdir(dir_path):
        return set()

    indices = set()
    for file_name in os.listdir(dir_path):
        if not file_name.startswith(prefix) or not file_name.endswith(suffix):
            continue
        number_text = file_name[len(prefix):len(file_name) - len(suffix)]
        if not number_text.isdigit():
            continue
        indices.add(int(number_text))
    return indices


def validate_indices_for_append(indices, expected_count, label):
    expected_indices = set(range(expected_count))
    if indices == expected_indices:
        return

    missing = sorted(expected_indices - indices)
    extra = sorted(indices - expected_indices)
    raise RuntimeError(
        f"{label} is not append-safe. "
        f"missing={missing[:10]} extra={extra[:10]} "
        f"expected_count={expected_count} actual_count={len(indices)}"
    )


def validate_hdf5_modalities_for_append(args, subtask_list, expected_count):
    data_type = args.get("data_type", {})
    require_depth = bool(data_type.get("depth", False))
    require_pointcloud = bool(data_type.get("pointcloud", False))
    if not require_depth and not require_pointcloud:
        return

    for sub_id in subtask_list:
        for episode_idx in range(expected_count):
            hdf5_path = os.path.join(
                get_subtask_data_dir(args, sub_id),
                f"episode{episode_idx}.hdf5",
            )
            with h5py.File(hdf5_path, "r") as file:
                if require_depth:
                    depth_keys = []
                    file.visit(
                        lambda name: depth_keys.append(name)
                        if name.lower().endswith("/depth") else None
                    )
                    if not depth_keys:
                        raise RuntimeError(
                            f"{hdf5_path} has no depth data; refusing append into mixed schema"
                        )
                if require_pointcloud:
                    if "pointcloud" not in file:
                        raise RuntimeError(
                            f"{hdf5_path} has no pointcloud dataset; refusing append"
                        )
                    shape = file["pointcloud"].shape
                    if len(shape) == 0 or 0 in shape:
                        raise RuntimeError(
                            f"{hdf5_path} has empty pointcloud shape={shape}; refusing append"
                        )


def validate_dataset_for_append(args, subtask_list, seed_list):
    expected_count = len(seed_list)
    if expected_count == 0:
        raise RuntimeError("seed.txt is empty; append mode needs existing data")

    for sub_id in subtask_list:
        legacy_traj_dir = os.path.join(get_subtask_root(args, sub_id), "traj_data")
        if os.path.isdir(legacy_traj_dir):
            legacy_files = [
                file_name for file_name in os.listdir(legacy_traj_dir)
                if file_name.endswith(".pkl")
            ]
            if legacy_files:
                raise RuntimeError(
                    f"data_subtask{sub_id}/traj_data contains legacy traj files; "
                    "rerun the dataset with the official _traj_data layout before append"
                )

        specs = [
            (get_subtask_traj_dir(args, sub_id), "episode", ".pkl", "_traj_data"),
            (get_subtask_data_dir(args, sub_id), "episode", ".hdf5", "data"),
            (get_subtask_video_dir(args, sub_id), "episode", ".mp4", "video"),
            (get_subtask_instruction_dir(args, sub_id), "episode", ".json", "instructions"),
        ]
        for dir_path, prefix, suffix, name in specs:
            indices = collect_numbered_episode_indices(dir_path, prefix, suffix)
            validate_indices_for_append(
                indices,
                expected_count,
                f"data_subtask{sub_id}/{name}",
            )

    info_file_path = os.path.join(args["save_path"], "scene_info.json")
    if not os.path.exists(info_file_path):
        raise RuntimeError(f"append mode requires existing scene_info.json: {info_file_path}")

    with open(info_file_path, "r", encoding="utf-8") as file:
        info_db = json.load(file)
    info_indices = set()
    for key in info_db:
        if not key.startswith("episode_"):
            continue
        number_text = key[len("episode_"):]
        if number_text.isdigit():
            info_indices.add(int(number_text))
    validate_indices_for_append(info_indices, expected_count, "scene_info.json")
    validate_hdf5_modalities_for_append(args, subtask_list, expected_count)


def ensure_episode_artifacts_absent(save_path, subtask_list, episode_idx):
    existing = [
        file_path
        for file_path in get_episode_artifact_paths(save_path, subtask_list, episode_idx)
        if os.path.exists(file_path)
    ]
    if existing:
        raise RuntimeError(
            f"Episode {episode_idx} already has artifacts; refusing to overwrite: {existing}"
        )


def load_replacement_manifest(manifest_path):
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    return manifest_path, manifest


def validate_replacement_manifest(args, manifest):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported replacement manifest version")
    if os.path.abspath(manifest["dataset_path"]) != os.path.abspath(args["save_path"]):
        raise ValueError("Replacement manifest belongs to a different dataset")
    if manifest["task_name"] != args["task_name"]:
        raise ValueError("Replacement manifest belongs to a different task")
    if manifest["task_config"] != args["task_config"]:
        raise ValueError("Replacement manifest belongs to a different task config")


def close_env_safely(task_env, clear_cache=False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except TypeError:
        task_env.close_env()
    except Exception as close_error:
        print(f"\033[93m[Warn] Failed to close env cleanly: {close_error}\033[0m")


def plan_replacement_candidate(task_env, args, subtask_list, episode_idx, seed):
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
    failed_subtask = None
    failed_reason = None

    for sub_id in subtask_list:
        task_env._setup_subtask_config(sub_id)
        reset_success = True
        try:
            run_subtask_motion_plan(task_env, sub_id)
            if sub_id != subtask_list[-1]:
                reset_success = reset_after_subtask(task_env, sub_id)
        except Exception as plan_error:
            print(
                f"\033[93m[Debug] Planning Error at seed {seed}: "
                f"Subtask {sub_id} failed. ({plan_error})\033[0m"
            )

        success = check_subtask_success(task_env, sub_id)
        update_subtask_state(task_env, sub_id, success)
        if not success:
            failed_subtask = sub_id
            failed_reason = "failed"
            break
        if not reset_success:
            failed_subtask = sub_id
            failed_reason = "reset failed"
            break

        save_subtask_traj_data(task_env, args, sub_id, str(episode_idx))

    close_env_safely(task_env)
    return failed_subtask, failed_reason


def render_replacement_candidate(task_env, args, subtask_list, episode_idx, seed):
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
    episode_info = get_scene_info(task_env)
    episode_info["episode_idx"] = episode_idx
    episode_info["subtask_info"] = {}

    for sub_id in subtask_list:
        task_env._setup_subtask_config(sub_id)
        traj_data = load_subtask_traj_data(args, sub_id, str(episode_idx))
        args["left_joint_path"] = traj_data.get("left_joint_path", [])
        args["right_joint_path"] = traj_data.get("right_joint_path", [])

        if len(args["left_joint_path"]) == 0 and len(args["right_joint_path"]) == 0:
            raise RuntimeError(f"Episode {episode_idx} Subtask {sub_id} trajectory is empty")

        task_env.set_path_lst(args)
        run_subtask_motion_plan(task_env, sub_id)
        subtask_scene_info = get_scene_info(task_env)
        subtask_params = subtask_scene_info.get("info", {})
        if not subtask_params:
            subtask_params = infer_instruction_params(task_env)
        episode_info["subtask_info"][str(sub_id)] = dict(subtask_params)
        if not episode_info.get("info") and subtask_params:
            episode_info["info"] = dict(subtask_params)

        if sub_id != subtask_list[-1]:
            reset_success = reset_after_subtask(task_env, sub_id)
            if not reset_success:
                raise RuntimeError(f"Episode {episode_idx} Subtask {sub_id} reset failed")

        success = check_subtask_success(task_env, sub_id)
        episode_info[f"subtask{sub_id}_success"] = success
        update_subtask_state(task_env, sub_id, success)
        save_subtask_data(task_env, args, sub_id)

        if not success:
            raise RuntimeError(f"Episode {episode_idx} Subtask {sub_id} replay failed")

    final_scene_info = get_scene_info(task_env)
    subtask_info = episode_info["subtask_info"]
    episode_info.update(final_scene_info)
    if not episode_info.get("info"):
        inferred_params = infer_instruction_params(task_env)
        if inferred_params:
            episode_info["info"] = dict(inferred_params)
    episode_info["episode_idx"] = episode_idx
    episode_info["subtask_info"] = subtask_info
    for sub_id in subtask_list:
        success_flags = getattr(task_env, "subtask_success", [])
        episode_info[f"subtask{sub_id}_success"] = bool(success_flags[sub_id - 1])
    save_episode_instructions(args, subtask_list, episode_idx, episode_info)

    close_env_safely(task_env, clear_cache=True)
    return episode_info


def stage_replacements(task_env, args, target_episodes, manifest_path=None, rejected=None):
    subtask_list = get_subtask_list(task_env)
    dataset_path = os.path.abspath(args["save_path"])
    rejected = rejected or []

    if manifest_path is None:
        seed_list = read_seed_list(dataset_path)
        if not seed_list:
            raise ValueError("seed.txt is empty")

        target_episodes = [int(episode_idx) for episode_idx in target_episodes]
        if len(target_episodes) != len(set(target_episodes)):
            raise ValueError("Replacement episode indices must be unique")
        if any(episode_idx < 0 or episode_idx >= len(seed_list) for episode_idx in target_episodes):
            raise ValueError("Replacement episode index is outside the existing dataset")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_root = os.path.join(dataset_path, ".replacement_staging", timestamp)
        stage_data_path = os.path.join(stage_root, "candidate_data")
        manifest_path = os.path.join(stage_root, "manifest.json")
        manifest = {
            "version": 1,
            "status": "staging",
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "dataset_path": dataset_path,
            "stage_data_path": stage_data_path,
            "target_episodes": target_episodes,
            "base_seed_list": seed_list,
            "next_seed": seed_list[-1] + 1,
            "candidates": {},
        }
    else:
        manifest_path, manifest = load_replacement_manifest(manifest_path)
        validate_replacement_manifest(args, manifest)
        if manifest["status"] not in ("staging", "ready"):
            raise ValueError(f"Cannot resume manifest with status {manifest['status']}")
        target_episodes = manifest["target_episodes"]
        stage_data_path = manifest["stage_data_path"]

    stage_args = args.copy()
    stage_args["save_path"] = stage_data_path
    ensure_subtask_dirs(stage_args, subtask_list)

    for episode_idx in rejected:
        episode_key = str(episode_idx)
        if episode_idx not in target_episodes:
            raise ValueError(f"Episode {episode_idx} is not a replacement target")
        clear_episode_artifacts(stage_data_path, subtask_list, episode_idx)
        manifest["candidates"].pop(episode_key, None)
        print(f"\033[93m[Rejected] Discarded staged replacement for Episode {episode_idx}\033[0m")

    manifest["status"] = "staging"
    atomic_write_json(manifest_path, manifest)

    for episode_idx in target_episodes:
        episode_key = str(episode_idx)
        if episode_key in manifest["candidates"]:
            continue

        while episode_key not in manifest["candidates"]:
            seed = manifest["next_seed"]
            manifest["next_seed"] = seed + 1
            atomic_write_json(manifest_path, manifest)
            clear_episode_artifacts(stage_data_path, subtask_list, episode_idx)

            print(
                f"\033[34m[Replacement Planning] Episode {episode_idx}, seed {seed}\033[0m"
            )
            try:
                stage_args["need_plan"] = True
                failed_subtask, failed_reason = plan_replacement_candidate(
                    task_env, stage_args, subtask_list, episode_idx, seed
                )
                if failed_subtask is not None:
                    reason = failed_reason or "planning failed"
                    print(
                        f"\033[91m[Rejected] seed {seed}, "
                        f"Subtask {failed_subtask} {reason}\033[0m"
                    )
                    clear_episode_artifacts(stage_data_path, subtask_list, episode_idx)
                    continue

                stage_args["need_plan"] = False
                stage_args["save_data"] = True
                stage_args["render_freq"] = 0
                episode_info = render_replacement_candidate(
                    task_env, stage_args, subtask_list, episode_idx, seed
                )
                manifest["candidates"][episode_key] = {
                    "seed": seed,
                    "scene_info": episode_info,
                }
                atomic_write_json(manifest_path, manifest)
                print(
                    f"\033[92m[Staged] Episode {episode_idx} replacement uses seed {seed}\033[0m"
                )
            except Exception as error:
                print(f"\033[91m[Rejected] seed {seed}: {error}\033[0m")
                close_env_safely(task_env, clear_cache=True)
                clear_episode_artifacts(stage_data_path, subtask_list, episode_idx)

    manifest["status"] = "ready"
    atomic_write_json(manifest_path, manifest)
    print(f"\033[92m[Ready For Review] {manifest_path}\033[0m")
    for episode_idx in target_episodes:
        video_path = os.path.join(
            stage_data_path, "data_subtask2", "video", f"episode{episode_idx}.mp4"
        )
        print(f"Review Episode {episode_idx}: {video_path}")


def copy_with_atomic_replace(source_path, target_path):
    tmp_path = f"{target_path}.replacement-tmp"
    shutil.copy2(source_path, tmp_path)
    os.replace(tmp_path, target_path)


def move_with_atomic_replace(source_path, target_path):
    tmp_path = f"{target_path}.episode-tmp"
    shutil.move(source_path, tmp_path)
    os.replace(tmp_path, target_path)


def get_episode_staging_path(args, episode_idx):
    return os.path.join(args["save_path"], ".episode_staging", f"episode{episode_idx}")


def prepare_episode_staging(args, subtask_list, episode_idx):
    staging_path = get_episode_staging_path(args, episode_idx)
    if os.path.exists(staging_path):
        shutil.rmtree(staging_path)
    staging_args = args.copy()
    staging_args["save_path"] = staging_path
    ensure_subtask_dirs(staging_args, subtask_list)
    return staging_path, staging_args


def ensure_episode_artifacts_readable(save_path, subtask_list, episode_idx):
    for sub_id in subtask_list:
        traj_path = os.path.join(
            save_path, f"data_subtask{sub_id}", "_traj_data", f"episode{episode_idx}.pkl"
        )
        data_path = os.path.join(
            save_path, f"data_subtask{sub_id}", "data", f"episode{episode_idx}.hdf5"
        )
        video_path = os.path.join(
            save_path, f"data_subtask{sub_id}", "video", f"episode{episode_idx}.mp4"
        )
        instruction_path = os.path.join(
            save_path, f"data_subtask{sub_id}", "instructions", f"episode{episode_idx}.json"
        )

        with open(traj_path, "rb") as file:
            pickle.load(file)

        with h5py.File(data_path, "r") as file:
            if len(file.keys()) == 0:
                raise RuntimeError(f"{data_path} is empty")

        video = cv2.VideoCapture(video_path)
        try:
            if not video.isOpened():
                raise RuntimeError(f"{video_path} cannot be opened")
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError(f"{video_path} has no frames")
        finally:
            video.release()

        with open(instruction_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload.get("seen"), list) or not isinstance(payload.get("unseen"), list):
            raise RuntimeError(f"{instruction_path} is not an official instruction json")


def render_episode_to_staging(task_env, args, subtask_list, episode_idx, seed):
    staging_path = get_episode_staging_path(args, episode_idx)

    staging_args = args.copy()
    staging_args["save_path"] = staging_path
    ensure_subtask_dirs(staging_args, subtask_list)

    for sub_id in subtask_list:
        source_traj = get_traj_file_path(args, sub_id, str(episode_idx))
        target_traj = get_traj_file_path(staging_args, sub_id, str(episode_idx))
        if not os.path.exists(target_traj):
            shutil.copy2(source_traj, target_traj)

    episode_info = render_replacement_candidate(
        task_env, staging_args, subtask_list, episode_idx, seed
    )
    ensure_episode_artifacts_readable(staging_path, subtask_list, episode_idx)
    return staging_path, episode_info


def commit_staged_episode(staging_path, args, subtask_list, episode_idx):
    ensure_episode_artifacts_absent(args["save_path"], subtask_list, episode_idx)
    for sub_id in subtask_list:
        staged_files = [
            (
                os.path.join(staging_path, f"data_subtask{sub_id}", "_traj_data", f"episode{episode_idx}.pkl"),
                os.path.join(get_subtask_traj_dir(args, sub_id), f"episode{episode_idx}.pkl"),
            ),
            (
                os.path.join(staging_path, f"data_subtask{sub_id}", "data", f"episode{episode_idx}.hdf5"),
                os.path.join(get_subtask_data_dir(args, sub_id), f"episode{episode_idx}.hdf5"),
            ),
            (
                os.path.join(staging_path, f"data_subtask{sub_id}", "video", f"episode{episode_idx}.mp4"),
                os.path.join(get_subtask_video_dir(args, sub_id), f"episode{episode_idx}.mp4"),
            ),
            (
                os.path.join(staging_path, f"data_subtask{sub_id}", "instructions", f"episode{episode_idx}.json"),
                os.path.join(get_subtask_instruction_dir(args, sub_id), f"episode{episode_idx}.json"),
            ),
        ]
        for source_path, target_path in staged_files:
            move_with_atomic_replace(source_path, target_path)

    shutil.rmtree(staging_path)
    staging_parent = os.path.dirname(staging_path)
    try:
        os.rmdir(staging_parent)
    except OSError:
        pass


def validate_committed_dataset(dataset_path, subtask_list, expected_count):
    for sub_id in subtask_list:
        expected_dirs = {
            "_traj_data": ".pkl",
            "data": ".hdf5",
            "video": ".mp4",
            "instructions": ".json",
        }
        for dir_name, suffix in expected_dirs.items():
            dir_path = os.path.join(dataset_path, f"data_subtask{sub_id}", dir_name)
            actual_count = len([
                file_name for file_name in os.listdir(dir_path)
                if file_name.endswith(suffix)
            ])
            if actual_count != expected_count:
                raise RuntimeError(
                    f"{dir_path} contains {actual_count} files, expected {expected_count}"
                )


def commit_replacements(task_env, args, manifest_path):
    manifest_path, manifest = load_replacement_manifest(manifest_path)
    validate_replacement_manifest(args, manifest)
    if manifest["status"] != "ready":
        raise ValueError(f"Cannot commit manifest with status {manifest['status']}")

    dataset_path = manifest["dataset_path"]
    stage_data_path = manifest["stage_data_path"]
    target_episodes = manifest["target_episodes"]
    subtask_list = get_subtask_list(task_env)
    seed_list = read_seed_list(dataset_path)

    if seed_list != manifest["base_seed_list"]:
        raise RuntimeError("seed.txt changed after staging; refusing to overwrite dataset")

    candidate_artifacts = []
    target_artifacts = []
    for episode_idx in target_episodes:
        episode_key = str(episode_idx)
        if episode_key not in manifest["candidates"]:
            raise RuntimeError(f"Missing staged candidate for Episode {episode_idx}")
        candidate_artifacts.extend(
            get_episode_artifact_paths(stage_data_path, subtask_list, episode_idx)
        )
        target_artifacts.extend(
            get_episode_artifact_paths(dataset_path, subtask_list, episode_idx)
        )

    for file_path in candidate_artifacts:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
    target_artifacts_existed = {
        file_path: os.path.isfile(file_path) for file_path in target_artifacts
    }

    scene_info_path = os.path.join(dataset_path, "scene_info.json")
    seed_file_path = os.path.join(dataset_path, "seed.txt")
    with open(scene_info_path, "r", encoding="utf-8") as file:
        scene_info = json.load(file)

    rollback_path = os.path.join(os.path.dirname(manifest_path), "rollback")
    if os.path.exists(rollback_path):
        shutil.rmtree(rollback_path)
    os.makedirs(rollback_path)

    try:
        for target_path in target_artifacts:
            if not target_artifacts_existed[target_path]:
                continue
            relative_path = os.path.relpath(target_path, dataset_path)
            rollback_file_path = os.path.join(rollback_path, relative_path)
            os.makedirs(os.path.dirname(rollback_file_path), exist_ok=True)
            shutil.copy2(target_path, rollback_file_path)
        shutil.copy2(seed_file_path, os.path.join(rollback_path, "seed.txt"))
        shutil.copy2(scene_info_path, os.path.join(rollback_path, "scene_info.json"))

        for source_path, target_path in zip(candidate_artifacts, target_artifacts):
            copy_with_atomic_replace(source_path, target_path)

        for episode_idx in target_episodes:
            episode_key = str(episode_idx)
            seed_list[episode_idx] = manifest["candidates"][episode_key]["seed"]
            scene_info[f"episode_{episode_idx}"] = manifest["candidates"][episode_key][
                "scene_info"
            ]

        atomic_write_text(seed_file_path, "".join(f"{seed} " for seed in seed_list))
        atomic_write_json(scene_info_path, scene_info)
        validate_committed_dataset(dataset_path, subtask_list, len(seed_list))
    except Exception:
        for target_path in target_artifacts:
            relative_path = os.path.relpath(target_path, dataset_path)
            rollback_file_path = os.path.join(rollback_path, relative_path)
            if os.path.isfile(rollback_file_path):
                copy_with_atomic_replace(rollback_file_path, target_path)
            elif os.path.exists(target_path):
                os.remove(target_path)
        copy_with_atomic_replace(os.path.join(rollback_path, "seed.txt"), seed_file_path)
        copy_with_atomic_replace(
            os.path.join(rollback_path, "scene_info.json"), scene_info_path
        )
        raise
    finally:
        if os.path.exists(rollback_path):
            shutil.rmtree(rollback_path)

    stage_root = os.path.dirname(manifest_path)
    shutil.rmtree(stage_root)
    staging_parent = os.path.dirname(stage_root)
    try:
        os.rmdir(staging_parent)
    except OSError:
        pass
    print(
        f"\033[92m[Committed] Replaced Episodes {target_episodes}; "
        "temporary staging data removed.\033[0m"
    )


def append_full_task_data(task_env, args, append_num=None):
    target_num = append_num if append_num is not None else args.get("episode_num", 5)
    if target_num <= 0:
        raise ValueError("--append-num must be greater than 0")

    subtask_list = get_subtask_list(task_env)
    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    print(f"Subtasks: {subtask_list}")
    print(f"Append Episodes: \033[34m{target_num}\033[0m")

    os.makedirs(args["save_path"], exist_ok=True)
    ensure_subtask_dirs(args, subtask_list)

    seed_file_path = os.path.join(args["save_path"], "seed.txt")
    info_file_path = os.path.join(args["save_path"], "scene_info.json")
    if not os.path.exists(seed_file_path):
        raise FileNotFoundError(f"append mode requires existing seed.txt: {seed_file_path}")

    seed_list = read_seed_list(args["save_path"])
    validate_dataset_for_append(args, subtask_list, seed_list)

    if os.path.exists(info_file_path):
        with open(info_file_path, "r", encoding="utf-8") as file:
            info_db = json.load(file)
    else:
        info_db = {}

    next_episode_idx = len(seed_list)
    epid = max(seed_list) + 1
    start_seed = epid
    appended_num = 0
    fail_num = 0

    print(
        f"\033[93m[Append Mode]\033[0m start_episode={next_episode_idx}, "
        f"start_seed={epid}"
    )

    while appended_num < target_num:
        episode_idx = next_episode_idx + appended_num
        seed = epid
        epid += 1
        staging_path = None

        try:
            ensure_episode_artifacts_absent(args["save_path"], subtask_list, episode_idx)
            staging_path, staging_args = prepare_episode_staging(
                args, subtask_list, episode_idx
            )
            print(
                f"\033[34m[Append Planning] Episode {episode_idx}, seed {seed}\033[0m"
            )

            staging_args["need_plan"] = True
            failed_subtask, failed_reason = plan_replacement_candidate(
                task_env, staging_args, subtask_list, episode_idx, seed
            )
            if failed_subtask is not None:
                reason = failed_reason or "failed"
                print(
                    f"\033[91msimulate data episode fail! "
                    f"(seed = {seed}, Subtask {failed_subtask} {reason})\033[0m"
                )
                if staging_path and os.path.exists(staging_path):
                    shutil.rmtree(staging_path)
                fail_num += 1
                continue

            print(f"simulate data episode {episode_idx} success! (seed = {seed})")

            args["need_plan"] = False
            args["save_data"] = True
            args["render_freq"] = 0

            print(
                f"\033[34m[Append Rendering] Episode {episode_idx}, seed {seed}\033[0m"
            )
            staging_path, episode_info = render_episode_to_staging(
                task_env, args, subtask_list, episode_idx, seed
            )
            commit_staged_episode(staging_path, args, subtask_list, episode_idx)

            seed_list.append(seed)
            info_db[f"episode_{episode_idx}"] = episode_info
            atomic_write_text(seed_file_path, "".join(f"{sed} " for sed in seed_list))
            atomic_write_json(info_file_path, info_db)

            appended_num += 1
            print(
                f"\033[92m[Append Saved] Episode {episode_idx} saved. "
                f"(seed = {seed})\033[0m"
            )
        except Exception as error:
            print(f"\033[91m[Append Rejected] seed={seed}: {error}\033[0m")
            close_env_safely(task_env, clear_cache=True)
            if staging_path and os.path.exists(staging_path):
                shutil.rmtree(staging_path)
            clear_episode_artifacts(args["save_path"], subtask_list, episode_idx)
            fail_num += 1

    validate_dataset_for_append(args, subtask_list, seed_list)
    print(
        f"\nAppend complete: {appended_num} new success / "
        f"{epid - start_seed} tries"
    )
    print(f"Total episodes: \033[92m{len(seed_list)}\033[0m  Failures: \033[91m{fail_num}\033[0m")


def main(task_name=None, task_config=None, cli_args=None):
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(
        CONFIGS_PATH, "_embodiment_config.yml")

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

    args["left_embodiment_config"] = get_embodiment_config(
        args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(
        args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(
            embodiment_type[0]) + "+" + str(embodiment_type[1])

    print("============= Config =============")
    print("\033[95mMessy Table:\033[0m " +
          str(args["domain_randomization"]["cluttered_table"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("==================================")

    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(
        args["task_name"]), args["task_config"])

    cli_args = cli_args or {}
    if cli_args.get("episode_num") is not None:
        args["episode_num"] = cli_args["episode_num"]

    if cli_args.get("commit_replacements"):
        commit_replacements(task, args, cli_args["commit_replacements"])
    elif cli_args.get("resume_replacements"):
        stage_replacements(
            task,
            args,
            target_episodes=None,
            manifest_path=cli_args["resume_replacements"],
            rejected=cli_args.get("reject_episodes"),
        )
    elif cli_args.get("stage_replacements"):
        stage_replacements(task, args, cli_args["stage_replacements"])
    elif cli_args.get("append_data"):
        append_full_task_data(task, args, cli_args.get("append_num"))
    else:
        run(task, args)


def run(TASK_ENV, args):
    TARGET_NUM = args.get("episode_num", 5)
    SUBTASK_LIST = get_subtask_list(TASK_ENV)

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    print(f"Subtasks: {SUBTASK_LIST}")

    os.makedirs(args["save_path"], exist_ok=True)
    ensure_subtask_dirs(args, SUBTASK_LIST)

    seed_file_path = os.path.join(args["save_path"], "seed.txt")
    info_file_path = os.path.join(args["save_path"], "scene_info.json")

    if args["use_seed"]:
        seed_list = read_seed_list(args["save_path"])[:TARGET_NUM]
        if len(seed_list) < TARGET_NUM:
            raise RuntimeError(
                f"seed.txt only contains {len(seed_list)} seeds, expected {TARGET_NUM}"
            )
        start_seed = None
    else:
        seed_list = []
        start_seed = 0

    info_db = {}
    fail_num = 0
    reset_fail_num = 0
    reset_fail_by_subtask = {sub_id: 0 for sub_id in SUBTASK_LIST}
    epid = start_seed if start_seed is not None else None
    tries = 0
    processed_count = 0

    print("\033[93m[One-by-One Data Generation]\033[0m")
    while True:
        if args["use_seed"]:
            if processed_count >= TARGET_NUM:
                break
        elif len(seed_list) >= TARGET_NUM:
            break

        if args["use_seed"]:
            episode_idx = processed_count
            seed = seed_list[episode_idx]
        else:
            episode_idx = len(seed_list)
            seed = epid
            epid += 1
            tries += 1
        staging_path = None

        try:
            ensure_episode_artifacts_absent(args["save_path"], SUBTASK_LIST, episode_idx)
            staging_path, staging_args = prepare_episode_staging(
                args, SUBTASK_LIST, episode_idx
            )
            staging_args["need_plan"] = True
            failed_subtask, failed_reason = plan_replacement_candidate(
                TASK_ENV, staging_args, SUBTASK_LIST, episode_idx, seed
            )
            if failed_subtask is not None:
                reason = failed_reason or "failed"
                print(
                    f"\033[91msimulate data episode fail! "
                    f"(seed = {seed}, Subtask {failed_subtask} {reason})\033[0m"
                )
                if reason == "reset failed":
                    reset_fail_num += 1
                    reset_fail_by_subtask[failed_subtask] = (
                        reset_fail_by_subtask.get(failed_subtask, 0) + 1
                    )
                if staging_path and os.path.exists(staging_path):
                    shutil.rmtree(staging_path)
                fail_num += 1
                if args["use_seed"]:
                    info_db[f"episode_{episode_idx}"] = {
                        f"subtask{failed_subtask}_success": False
                    }
                    processed_count += 1
                continue

            print(f"simulate data episode {episode_idx} success! (seed = {seed})")

            if args.get("collect_data", True):
                print(f"\033[34mRendering Video for Episode {episode_idx}\033[0m")
                args["need_plan"] = False
                args["save_data"] = True
                args["render_freq"] = 0
                staging_path, episode_info = render_episode_to_staging(
                    TASK_ENV, args, SUBTASK_LIST, episode_idx, seed
                )
                commit_staged_episode(staging_path, args, SUBTASK_LIST, episode_idx)
                info_db[f"episode_{episode_idx}"] = episode_info
                atomic_write_json(info_file_path, info_db)
                print(f"\033[92m[Saved] Episode {episode_idx} saved. (seed = {seed})\033[0m")

            if not args["use_seed"]:
                seed_list.append(seed)
                atomic_write_text(seed_file_path, "".join(f"{sed} " for sed in seed_list))
            else:
                processed_count += 1
        except Exception as e:
            print(f"\033[91mEpisode seed={seed} 彻底崩溃: {e}\033[0m")
            if "reset failed" in str(e):
                reset_fail_num += 1
            close_env_safely(TASK_ENV, clear_cache=True)
            if staging_path and os.path.exists(staging_path):
                shutil.rmtree(staging_path)
            clear_episode_artifacts(args["save_path"], SUBTASK_LIST, episode_idx)
            fail_num += 1
            if args["use_seed"]:
                processed_count += 1
            continue

    if args["use_seed"]:
        atomic_write_text(seed_file_path, "".join(f"{sed} " for sed in seed_list))
        tries = len(seed_list)

    total_tries = tries if tries else len(seed_list)
    success_rate = len(seed_list) / total_tries * 100 if total_tries else 0
    print(f"\nComplete simulation: {len(seed_list)} success / {total_tries} tries")
    print(
        f"成功率: \033[92m{success_rate:.1f}%\033[0m  "
        f"(失败 \033[91m{fail_num}\033[0m 次)\n"
    )
    reset_fail_detail = ", ".join(
        f"Subtask {sub_id}: {count}"
        for sub_id, count in sorted(reset_fail_by_subtask.items())
        if count
    ) or "none"
    print(
        f"Reset failures: \033[91m{reset_fail_num}\033[0m  "
        f"({reset_fail_detail})\n"
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    arg_parser = ArgumentParser()
    arg_parser.add_argument("task_name", type=str)
    arg_parser.add_argument("task_config", type=str)
    replacement_mode = arg_parser.add_mutually_exclusive_group()
    replacement_mode.add_argument("--stage-replacements", nargs="+", type=int)
    replacement_mode.add_argument("--resume-replacements", type=str)
    replacement_mode.add_argument("--commit-replacements", type=str)
    replacement_mode.add_argument("--append-data", action="store_true")
    arg_parser.add_argument("--reject-episodes", nargs="*", type=int, default=[])
    arg_parser.add_argument("--append-num", type=int, default=None)
    arg_parser.add_argument("--episode-num", type=int, default=None)
    parser = arg_parser.parse_args()
    if parser.append_num is not None and not parser.append_data:
        arg_parser.error("--append-num can only be used with --append-data")

    main(
        task_name=parser.task_name,
        task_config=parser.task_config,
        cli_args=vars(parser),
    )
