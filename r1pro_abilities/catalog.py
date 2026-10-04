"""七类 Ability 的运行角色和本地 Task 注册清单。"""

from enum import StrEnum


class AbilityRole(StrEnum):
    NAVIGATION = "navigation"
    MANIPULATOR_MOTION = "manipulator_motion"
    END_EFFECTOR = "end_effector"
    ROBOT_STATE = "robot_state"
    SENSOR_CAPTURE = "sensor_capture"
    OBJECT_PERCEPTION = "object_perception"
    GRASP_PLANNING = "grasp_planning"


TASKS: dict[AbilityRole, tuple[str, ...]] = {
    AbilityRole.NAVIGATION: ("PlanRoute", "FollowRoute", "VerifyArrival"),
    AbilityRole.MANIPULATOR_MOTION: (
        "MoveEndEffector",
        "FollowWaypoints",
        "LiftHeldObject",
        "MoveToPosture",
    ),
    AbilityRole.END_EFFECTOR: (
        "SetOpening",
        "CloseUntilContact",
        "Release",
        "HoldObject",
    ),
    AbilityRole.ROBOT_STATE: ("GetRobotState", "VerifyToolLoad"),
    AbilityRole.SENSOR_CAPTURE: ("CaptureRGBD",),
    AbilityRole.OBJECT_PERCEPTION: (
        "LocateObject",
        "VerifyPregrasp",
        "VerifyGrasp",
        "ObservePlacementTarget",
        "VerifyPlacement",
    ),
    AbilityRole.GRASP_PLANNING: ("GenerateCandidates", "PlanTransportPosture"),
}
