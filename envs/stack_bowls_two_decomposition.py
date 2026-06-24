from .stack_bowls_two import stack_bowls_two
from .utils import ArmTag
import numpy as np


class stack_bowls_two_decomposition(stack_bowls_two):
    def __init__(self, **args):
        super().__init__(**args)
        self.current_subtask = 0
        self.subtask_done = [False, False]
        self.subtask_success = [False, False]

    def play_once(self):
        self._execute_subtask(1)
        if not self.subtask_success[0]:
            self.info["info"] = {
                "subtask1_success": False,
                "subtask2_success": False,
                "failed_subtask": 1,
                "failed_reason": "subtask1 failed: bottom bowl was not placed at the base target",
            }
            return self.info

        self._reset_after_subtask(subtask_id=1)

        self._execute_subtask(2)
        if not self.subtask_success[1]:
            print("[FAIL] Subtask 2 failed: bowls were not stacked")

        self.info["info"] = {
            "subtask1_success": self.subtask_success[0],
            "subtask2_success": self.subtask_success[1],
            "failed_subtask": None if self.check_success() else 2,
            "{A}": "002_bowl/base3",
            "{B}": "002_bowl/base3",
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
        self._clear_path_cache()

        # Each decomposed phase starts after the arms have been reset.
        self.las_arm = None

    def _clear_path_cache(self):
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0

    def _subtask1_motion_plan(self):
        arm_tag1 = self.move_bowl(self.bowl1, self.bowl1_target_pose)
        arm_tag2 = ArmTag("left" if self.bowl2.get_pose().p[0] < 0 else "right")
        self.info["info"] = {
            "{A}": "002_bowl/base3",
            "{B}": "002_bowl/base3",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
        }

    def _subtask2_motion_plan(self):
        arm_tag2 = self.move_bowl(
            self.bowl2, self.bowl1.get_pose().p + [0, 0, 0.05]
        )
        self.info["info"] = {
            "{A}": "002_bowl/base3",
            "{B}": "002_bowl/base3",
            "{b}": str(arm_tag2),
        }

    def _check_subtask1_success(self):
        bowl1_pose = self.bowl1.get_pose().p
        target_pose = np.array([0, -0.1, 0.74 + self.table_z_bias])
        eps = np.array([0.03, 0.03, 0.02])

        return (
            np.all(abs(bowl1_pose[:3] - target_pose) < eps)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def _check_subtask2_success(self):
        return super().check_success()

    def check_success(self):
        return self.subtask_success[0] and self.subtask_success[1]

    def _reset_after_subtask(self, subtask_id=None):
        if subtask_id == 2:
            return True

        left_reset_success = self.move(self.back_to_origin(arm_tag="left"))
        right_reset_success = self.move(self.back_to_origin(arm_tag="right"))
        return bool(left_reset_success and right_reset_success and self.plan_success)
