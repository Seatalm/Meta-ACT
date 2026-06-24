from .put_object_cabinet import put_object_cabinet
import numpy as np


class put_object_cabinet_decomposition(put_object_cabinet):
    DRAWER_OPEN_MIN_DISPLACEMENT = 0.14
    DRAWER_ARM_MIN_HANDLE_DISTANCE = 0.05

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
                "failed_reason": "subtask1 failed: cabinet drawer was not opened",
            }
            return self.info

        self._execute_subtask(2)
        if not self.subtask_success[1]:
            print("[FAIL] Subtask 2 failed: object was not placed into the cabinet")

        self.info["info"] = {
            "subtask1_success": self.subtask_success[0],
            "subtask2_success": self.subtask_success[1],
            "failed_subtask": None if self.check_success() else 2,
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": "036_cabinet/base0",
            "{a}": str(self.arm_tag),
            "{b}": str(self.arm_tag.opposite),
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

        if subtask_id == 1:
            self.arm_tag = self._get_object_arm_tag()
            self.origin_z = self.object.get_pose().p[2]
            self.drawer_closed_fp = np.array(
                self.cabinet.get_functional_point(0)[:3])
        elif not hasattr(self, "arm_tag"):
            self.arm_tag = self._get_object_arm_tag()
        if not hasattr(self, "origin_z"):
            self.origin_z = self.object.get_pose().p[2]

    def _clear_path_cache(self):
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0

    def _get_object_arm_tag(self):
        from .utils import ArmTag

        return ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

    def _subtask1_motion_plan(self):
        self.arm_tag = self._get_object_arm_tag()
        self.origin_z = self.object.get_pose().p[2]
        self.drawer_closed_fp = np.array(
            self.cabinet.get_functional_point(0)[:3])
        self.target_pose = self.cabinet.get_functional_point(0)

        self.move(
            self.grasp_actor(
                self.cabinet,
                arm_tag=self.arm_tag.opposite,
                pre_grasp_dis=0.05,
            )
        )
        if not self.plan_success:
            return

        for _ in range(4):
            self.move(self.move_by_displacement(
                arm_tag=self.arm_tag.opposite, y=-0.04))
            if not self.plan_success:
                return

        self.move(self.open_gripper(arm_tag=self.arm_tag.opposite))
        if not self.plan_success:
            print("[RETREAT_TEST] subtask1 open_gripper failed")
            return

        self.move(self.move_by_displacement(
            arm_tag=self.arm_tag.opposite, y=-0.02, z=0.05))
        if not self.plan_success:
            print("[RETREAT_TEST] subtask1 retreat failed")
            return

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": "036_cabinet/base0",
            "{a}": str(self.arm_tag),
            "{b}": str(self.arm_tag.opposite),
        }

    def _subtask2_motion_plan(self):
        self.arm_tag = self._get_object_arm_tag()

        self.move(
            self.grasp_actor(
                self.object,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.1,
            )
        )
        if not self.plan_success:
            return

        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.15))
        if not self.plan_success:
            return

        target_pose = self.cabinet.get_functional_point(0)
        self.move(
            self.place_actor(
                self.object,
                arm_tag=self.arm_tag,
                target_pose=target_pose,
                pre_dis=0.13,
                dis=0.06,
            )
        )
        if not self.plan_success:
            return

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": "036_cabinet/base0",
            "{a}": str(self.arm_tag),
            "{b}": str(self.arm_tag.opposite),
        }

    def _drawer_arm_holding_cabinet(self, drawer_arm_tag):
        drawer_arm_close = (
            self.is_left_gripper_close()
            if drawer_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        if not drawer_arm_close:
            return False

        drawer_gripper_names = (
            {"fl_link7", "fl_link8"}
            if drawer_arm_tag == "left"
            else {"fr_link7", "fr_link8"}
        )
        cabinet_link_names = set(self.cabinet.link_dict.keys())
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if name0 in drawer_gripper_names and name1 in cabinet_link_names:
                return True
            if name1 in drawer_gripper_names and name0 in cabinet_link_names:
                return True
        return False

    def _check_subtask1_success(self):
        if not hasattr(self, "drawer_closed_fp") or not hasattr(self, "arm_tag"):
            return False

        current_fp = np.array(self.cabinet.get_functional_point(0)[:3])
        drawer_delta = self.drawer_closed_fp[1] - current_fp[1]
        success = drawer_delta > self.DRAWER_OPEN_MIN_DISPLACEMENT
        if not success:
            print(
                "[HOLD_DRAWER_TEST] subtask1 check failed: "
                f"drawer_delta={drawer_delta:.4f}, "
                f"min_delta={self.DRAWER_OPEN_MIN_DISPLACEMENT:.4f}"
            )
        return success

    def _check_subtask2_success(self):
        return super().check_success()

    def check_success(self):
        return self.subtask_success[0] and self.subtask_success[1]

    def _reset_after_subtask(self, subtask_id=None):
        if subtask_id == 2:
            return True

        if subtask_id == 1 and hasattr(self, "arm_tag"):
            drawer_arm_tag = self.arm_tag.opposite
            drawer_arm_open = (
                self.is_left_gripper_open()
                if drawer_arm_tag == "left"
                else self.is_right_gripper_open()
            )
            if drawer_arm_open:
                self.move(self.close_gripper(arm_tag=drawer_arm_tag))
                if not self.plan_success:
                    return False
            return self._drawer_arm_holding_cabinet(drawer_arm_tag)

        return bool(self.plan_success)
