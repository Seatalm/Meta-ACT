import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from argparse import Namespace
import base64
import hashlib
import json
from pathlib import Path


def _resize_rgb(observation, camera_name):
    return cv2.resize(
        observation["observation"][camera_name]["rgb"],
        (640, 480),
        interpolation=cv2.INTER_LINEAR,
    )


def _png_base64_from_rgb(image):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode image as png")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload_id(task_name, eval_test_index, take_action_cnt, mirror_counter):
    task_name = (task_name or "unknown_task").replace("/", "-")
    return (
        f"robotwin-{task_name}-test{int(eval_test_index):03d}-"
        f"step{int(take_action_cnt):04d}-mirror{int(mirror_counter):05d}"
    )


def _mirror_root_from_env():
    return (os.environ.get("ROBOTWIN_OPENCLAW_MIRROR_DIR") or "").strip()


def _mirror_every_from_env(default):
    raw = (os.environ.get("ROBOTWIN_OPENCLAW_MIRROR_EVERY") or str(default)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _maybe_init_openclaw_bridge(model):
    if getattr(model, "_openclaw_bridge_ready", False):
        return
    mirror_root = _mirror_root_from_env()
    model.openclaw_mirror_dir = mirror_root
    model.openclaw_mirror_every = (
        _mirror_every_from_env(1 if mirror_root else 0) if mirror_root else 0
    )
    model.openclaw_mirror_counter = 0
    model._openclaw_bridge_ready = True


def _maybe_export_openclaw_payload(task_env, model, observation, obs_for_act):
    _maybe_init_openclaw_bridge(model)
    mirror_root = getattr(model, "openclaw_mirror_dir", "")
    mirror_every = int(getattr(model, "openclaw_mirror_every", 0) or 0)
    mirror_counter = int(getattr(model, "openclaw_mirror_counter", 0) or 0)

    if not mirror_root or mirror_every <= 0:
        return
    if mirror_counter % mirror_every != 0:
        model.openclaw_mirror_counter = mirror_counter + 1
        return

    pending_dir = Path(mirror_root) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    head_cam = _resize_rgb(observation, "head_camera")
    left_cam = _resize_rgb(observation, "left_camera")
    right_cam = _resize_rgb(observation, "right_camera")
    instruction_text = task_env.get_instruction()
    qpos = list(obs_for_act["qpos"])

    payload = {
        "payload_version": 1,
        "source": "robotwin_act_deploy",
        "task_name": getattr(model, "task_name", None),
        "task_config": getattr(model, "task_config", None),
        "ckpt_setting": getattr(model, "ckpt_setting", None),
        "instruction_type": getattr(model, "instruction_type", None),
        "instruction_text": instruction_text,
        "eval_test_index": getattr(task_env, "test_num", None),
        "take_action_cnt_before_eval": getattr(task_env, "take_action_cnt", None),
        "policy_t": getattr(model, "t", None),
        "qpos_for_act": qpos,
        "head_cam_png_base64": _png_base64_from_rgb(head_cam),
        "left_cam_png_base64": _png_base64_from_rgb(left_cam),
        "right_cam_png_base64": _png_base64_from_rgb(right_cam),
        "camera_order": ["head_cam", "left_cam", "right_cam"],
    }
    payload["payload_id"] = _payload_id(
        payload["task_name"],
        payload["eval_test_index"],
        payload["take_action_cnt_before_eval"],
        mirror_counter,
    )
    payload["head_cam_sha256"] = hashlib.sha256(
        base64.b64decode(payload["head_cam_png_base64"])
    ).hexdigest()
    payload["left_cam_sha256"] = hashlib.sha256(
        base64.b64decode(payload["left_cam_png_base64"])
    ).hexdigest()
    payload["right_cam_sha256"] = hashlib.sha256(
        base64.b64decode(payload["right_cam_png_base64"])
    ).hexdigest()
    payload["act_input_signature_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "task_name": payload["task_name"],
                "qpos_for_act": payload["qpos_for_act"],
                "camera_hashes": [
                    payload["head_cam_sha256"],
                    payload["left_cam_sha256"],
                    payload["right_cam_sha256"],
                ],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    out_path = pending_dir / f"{payload['payload_id']}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    model.openclaw_mirror_counter = mirror_counter + 1


def encode_obs(observation):
    head_cam = _resize_rgb(observation, "head_camera")
    left_cam = _resize_rgb(observation, "left_camera")
    right_cam = _resize_rgb(observation, "right_camera")
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
    qpos = (
        observation["joint_action"]["left_arm"]
        + [observation["joint_action"]["left_gripper"]]
        + observation["joint_action"]["right_arm"]
        + [observation["joint_action"]["right_gripper"]]
    )
    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }


def get_model(usr_args):
    model = ACT(usr_args, Namespace(**usr_args))
    model.task_name = usr_args.get("task_name")
    model.task_config = usr_args.get("task_config")
    model.ckpt_setting = usr_args.get("ckpt_setting")
    model.instruction_type = usr_args.get("instruction_type")
    _maybe_init_openclaw_bridge(model)
    return model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    _maybe_export_openclaw_payload(TASK_ENV, model, observation, obs)
    # instruction = TASK_ENV.get_instruction()

    # Get action from model
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros(
            [
                model.max_timesteps,
                model.max_timesteps + model.num_queries,
                model.state_dim,
            ]
        ).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
