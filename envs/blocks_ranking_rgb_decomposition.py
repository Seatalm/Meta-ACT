from .blocks_ranking_rgb import blocks_ranking_rgb
from .utils import ArmTag
import numpy as np


class blocks_ranking_rgb_decomposition(blocks_ranking_rgb):
    def load_actors(self):
        super().load_actors()

        # ACT cannot infer a widely sampled placement target without a
        # corresponding visual cue. Keep target diversity while narrowing the
        # hidden y variation used by the decomposed task.
        target_y = np.random.uniform(-0.17, -0.13)
        self.block1_target_pose[:2] = [np.random.uniform(-0.09, -0.08), target_y]
        self.block2_target_pose[:2] = [np.random.uniform(-0.01, 0.01), target_y]
        self.block3_target_pose[:2] = [np.random.uniform(0.08, 0.09), target_y]

    def __init__(self, **args):
        super().__init__(**args)
        self.current_subtask = 0
        self.subtask_done = [False, False, False]
        self.subtask_success = [False, False, False]

    def play_once(self):
        self._execute_subtask(1)
        if not self.subtask_success[0]:
            self.info["info"] = {
                "subtask1_success": False,
                "subtask2_success": False,
                "subtask3_success": False,
                "failed_subtask": 1,
                "failed_reason": "subtask1 failed: red block was not placed at its target",
            }
            return self.info

        self._reset_after_subtask(subtask_id=1)

        self._execute_subtask(2)
        if not self.subtask_success[1]:
            self.info["info"] = {
                "subtask1_success": self.subtask_success[0],
                "subtask2_success": False,
                "subtask3_success": False,
                "failed_subtask": 2,
                "failed_reason": "subtask2 failed: green block was not placed after the red block",
            }
            return self.info

        self._reset_after_subtask(subtask_id=2)

        self._execute_subtask(3)
        if not self.subtask_success[2]:
            print("[FAIL] Subtask 3 failed: RGB block ranking is incomplete")

        self.info["info"] = {
            "subtask1_success": self.subtask_success[0],
            "subtask2_success": self.subtask_success[1],
            "subtask3_success": self.subtask_success[2],
            "failed_subtask": None if self.check_success() else 3,
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
        }
        return self.info

    def _execute_subtask(self, subtask_id):
        self.current_subtask = subtask_id
        self.subtask_done[subtask_id - 1] = False
        self.subtask_success[subtask_id - 1] = False

        self._setup_subtask_config(subtask_id)
        getattr(self, f"_subtask{subtask_id}_motion_plan")()

        self.subtask_success[subtask_id - 1] = getattr(
            self, f"_check_subtask{subtask_id}_success"
        )()
        self.subtask_done[subtask_id - 1] = True

    def _setup_subtask_config(self, subtask_id):
        self.current_subtask = subtask_id
        self.plan_success = True
        self.FRAME_IDX = 0
        self.last_gripper = None
        self._clear_path_cache()

    def _clear_path_cache(self):
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0

    def _subtask1_motion_plan(self):
        arm_tag1 = self.pick_and_place_block(
            self.block1, self.block1_target_pose)
        arm_tag2 = str(
            ArmTag("left" if self.block2.get_pose().p[0] < 0 else "right"))
        arm_tag3 = str(
            ArmTag("left" if self.block3.get_pose().p[0] < 0 else "right"))
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }

    def _subtask2_motion_plan(self):
        arm_tag2 = self.pick_and_place_block(
            self.block2, self.block2_target_pose)
        arm_tag3 = str(
            ArmTag("left" if self.block3.get_pose().p[0] < 0 else "right"))
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }

    def _subtask3_motion_plan(self):
        arm_tag3 = self.pick_and_place_block(
            self.block3, self.block3_target_pose)
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{c}": arm_tag3,
        }

    def _block_at_target(self, block, target_pose):
        block_pose = block.get_pose().p
        return np.all(abs(block_pose[:2] - np.array(target_pose[:2])) < np.array([0.05, 0.05]))

    def get_subtask_debug_info(self, subtask_id):
        def block_debug(block, target_pose):
            actual_xy = np.array(block.get_pose().p[:2])
            target_xy = np.array(target_pose[:2])
            return {
                "actual_xy": actual_xy.tolist(),
                "target_xy": target_xy.tolist(),
                "abs_error_xy": np.abs(actual_xy - target_xy).tolist(),
            }

        return {
            "subtask_id": subtask_id,
            "block1_red": block_debug(self.block1, self.block1_target_pose),
            "block2_green": block_debug(self.block2, self.block2_target_pose),
            "block3_blue": block_debug(self.block3, self.block3_target_pose),
        }

    def _check_subtask1_success(self):
        print(f"{self.block1.get_pose().p=}, {self.block1_target_pose=}")
        return (
            self._block_at_target(self.block1, self.block1_target_pose)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def _check_subtask2_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        eps = [0.13, 0.03]

        return (
            self._block_at_target(self.block1, self.block1_target_pose)
            and self._block_at_target(self.block2, self.block2_target_pose)
            and np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
            and block1_pose[0] < block2_pose[0]
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def _check_subtask3_success(self):
        return super().check_success()

    def check_success(self):
        return self.subtask_success[0] and self.subtask_success[1] and self.subtask_success[2]

    def _reset_after_subtask(self, subtask_id=None):
        if subtask_id == 3:
            return True

        left_reset_success = self.move(self.back_to_origin(arm_tag="left"))
        right_reset_success = self.move(self.back_to_origin(arm_tag="right"))
        return bool(left_reset_success and right_reset_success and self.plan_success)
