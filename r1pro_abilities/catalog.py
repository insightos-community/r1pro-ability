# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
