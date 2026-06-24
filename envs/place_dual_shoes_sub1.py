from ._base_task import Base_Task
from .utils import *
import math
import sapien
from ._GLOBAL_CONFIGS import *


class place_dual_shoes_sub1(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(table_height_bias=-0.1, **kwags)

    def load_actors(self):
        self.shoe_box = create_actor(
            self,
            pose=sapien.Pose([0, -0.13, 0.74], [0.5, 0.5, -0.5, -0.5]),
            modelname="007_shoe-box",
            convex=True,
            is_static=True,
        )

        shoe_id = np.random.choice([i for i in range(10)])
        self.shoe_id = shoe_id

        # left shoe
        shoes_pose = rand_pose(
            xlim=[-0.3, -0.2],
            ylim=[-0.1, 0.05],
            zlim=[0.741],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[-0.3, -0.2],
                ylim=[-0.1, 0.05],
                zlim=[0.741],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

        self.left_shoe = create_actor(
            self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=shoe_id,
        )

        # right shoe
        shoes_pose = rand_pose(
            xlim=[0.2, 0.3],
            ylim=[-0.1, 0.05],
            zlim=[0.741],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[0.2, 0.3],
                ylim=[-0.1, 0.05],
                zlim=[0.741],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

        self.right_shoe = create_actor(
            self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=shoe_id,
        )

        self.add_prohibit_area(self.left_shoe, padding=0.02)
        self.add_prohibit_area(self.right_shoe, padding=0.02)
        self.prohibited_area.append([-0.15, -0.25, 0.15, 0.01])
        self.right_shoe_middle_pose = [0.35, -0.05, 0.79, 0, 1, 0, 0]

    def play_once(self):
        left_arm_tag = ArmTag("left")
        # Only grasp and place the left shoe. Keep the right shoe untouched.
        self.move(
            self.grasp_actor(self.left_shoe, arm_tag=left_arm_tag, pre_grasp_dis=0.1),
        )
        # Lift the left shoe up only.
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.15),
        )
        # Place the left shoe into the shoe box target point.
        left_target = self.shoe_box.get_functional_point(0)
        left_place_pose = self.place_actor(
            self.left_shoe,
            target_pose=left_target,
            arm_tag=left_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        self.move(left_place_pose)
        if not self.plan_success:
            return self.info

        self.move(self.back_to_origin(left_arm_tag))

        self.delay(3)

        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": f"007_shoe-box/base0",
        }
        return self.info

    def check_success(self):
        left_shoe_pose_p = np.array(self.left_shoe.get_pose().p)
        left_shoe_pose_q = np.array(self.left_shoe.get_pose().q)
        if left_shoe_pose_q[0] < 0:
            left_shoe_pose_q *= -1
        target_pose_p = np.array([0, -0.13])
        target_pose_q = np.array([0.5, 0.5, -0.5, -0.5])
        eps = np.array([0.05, 0.05, 0.07, 0.07, 0.07, 0.07])
        return (np.all(abs(left_shoe_pose_p[:2] - (target_pose_p - [0, 0.04])) < eps[:2])
                and np.all(abs(left_shoe_pose_q - target_pose_q) < eps[-4:])
                and abs(left_shoe_pose_p[2] - (self.shoe_box.get_pose().p[2] + 0.01)) < 0.03
                and self.is_left_gripper_open())
