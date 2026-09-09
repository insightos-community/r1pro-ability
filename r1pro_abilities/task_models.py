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

"""Ability Task 的严格语义输入模型。

这些模型与 Robot Skill Action schema 对齐，但 Ability 不依赖 Skill 代码。厂商连接、
Topic、固件和模型路径只能来自部署配置，不能出现在 Task 输入中。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    invocation_id: str | None = None
    robot_id: str | None = None


class PoseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    frame_id: str
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4
    )
    observed_at: str | None = None
    revision: str | None = None


class NavigationTargetInput(BaseModel):
    """Robot Agent 已确定的导航意图，不是 Semantic Map 执行快照。

    pose 只作为路线规划目标；Navigation Provider 必须使用当前后端的可导航
    状态规划并在到达后重新观测。Map 的来源、generation 和 revision 留在
    Workflow/Task 追踪中，不进入 Ability 的物理执行协议。
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    target_ref: str
    pose: PoseInput
    constraints: dict[str, str | float | int | bool] = Field(default_factory=dict)


class PlanRouteInput(StrictInput):
    target: NavigationTargetInput
    navigation_purpose: Literal["approach_grasp", "carry_to_place", "transit"]
    maximum_speed_mps: float = Field(gt=0)
    minimum_clearance_m: float = Field(gt=0)
    carrying_object: dict[str, Any] | None = None


class FollowRouteInput(StrictInput):
    route_ref: str | None = None
    navigation_purpose: (
        Literal["approach_grasp", "carry_to_place", "transit"] | None
    ) = None
    maximum_speed_mps: float | None = Field(default=None, gt=0)
    minimum_clearance_m: float | None = Field(default=None, gt=0)
    carrying_object: dict[str, Any] | None = None
    reason: str | None = None
    mode: Literal["safe", "immediate"] | None = None

    @model_validator(mode="after")
    def validate_variant(self) -> "FollowRouteInput":
        execute = all(
            value is not None
            for value in (
                self.route_ref,
                self.navigation_purpose,
                self.maximum_speed_mps,
                self.minimum_clearance_m,
            )
        )
        stop = self.reason is not None and self.mode is not None
        if execute == stop:
            raise ValueError("FollowRoute 必须且只能匹配路线执行或安全停止之一")
        return self


class VerifyArrivalInput(StrictInput):
    target: NavigationTargetInput
    navigation_purpose: Literal["approach_grasp", "carry_to_place", "transit"]
    arrival_radius_m: float = Field(gt=0)
    require_visual_confirmation: bool
    carrying_object: dict[str, Any] | None = None


class EndEffectorTargetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool_ref: str
    target_pose: PoseInput


class ToolCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool_ref: str
    target_position_m: float = Field(ge=0)
    maximum_force_n: float = Field(gt=0)
    hold: bool = False


def _unique_tool_refs(values: list[Any], field: str = "tool_ref") -> list[Any]:
    refs = [getattr(item, field) for item in values]
    if len(refs) != len(set(refs)):
        raise ValueError("同一个 Action 不能重复声明工具")
    return values


class WaypointInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Literal["preplace", "release", "retreat"]
    pose: PoseInput


class MoveEndEffectorInput(StrictInput):
    targets: list[EndEffectorTargetInput] = Field(min_length=1, max_length=2)
    coordination: Literal["synchronized"] = "synchronized"
    purpose: Literal[
        "clearance",
        "transfer",
        "alignment",
        "pregrasp",
        "insert",
        "seat",
        "extract",
        "transport",
        "place",
        "unseat",
        "disengage",
        "retreat",
    ]
    object_ref: str | None = None
    candidate_id: str | None = None
    target_ref: str | None = None
    target_revision: str
    expected_object_pose: PoseInput | None = None
    maximum_speed_mps: float = Field(gt=0)
    max_contact_force_n: float | None = Field(default=None, gt=0)
    required_contact_tools: list[str] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_tools(self) -> "MoveEndEffectorInput":
        _unique_tool_refs(self.targets)
        if len(self.required_contact_tools) != len(set(self.required_contact_tools)):
            raise ValueError("required_contact_tools 不能重复声明工具")
        if self.purpose == "place" and (
            self.object_ref is None
            or self.target_ref is None
            or self.expected_object_pose is None
        ):
            raise ValueError(
                "place 运动必须声明物体、目标和本次实时观测得到的目标物体位姿"
            )
        return self


class FollowWaypointsInput(StrictInput):
    targets: list[EndEffectorTargetInput] = Field(min_length=1, max_length=2)
    coordination: Literal["synchronized"] = "synchronized"
    maximum_speed_mps: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_tools(self) -> "FollowWaypointsInput":
        _unique_tool_refs(self.targets)
        return self


class LiftHeldObjectInput(StrictInput):
    object_ref: str
    tools: list[str] = Field(min_length=1, max_length=2)
    candidate_id: str
    distance_m: float = Field(gt=0)
    maximum_speed_mps: float = Field(gt=0)


class MoveToPostureInput(StrictInput):
    posture: str = Field(min_length=1)


class SetOpeningInput(StrictInput):
    tools: list[ToolCommandInput] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_tools(self) -> "SetOpeningInput":
        _unique_tool_refs(self.tools)
        return self


class CloseUntilContactInput(StrictInput):
    object_ref: str
    tools: list[ToolCommandInput] = Field(min_length=1, max_length=2)
    candidate_id: str
    grasp_pose: PoseInput

    @model_validator(mode="after")
    def validate_tools(self) -> "CloseUntilContactInput":
        _unique_tool_refs(self.tools)
        return self


class ReleaseInput(StrictInput):
    object_ref: str
    tools: list[ToolCommandInput] = Field(min_length=1, max_length=2)
    target_ref: str
    target_revision: str

    @model_validator(mode="after")
    def validate_tools(self) -> "ReleaseInput":
        _unique_tool_refs(self.tools)
        return self


class HoldObjectInput(StrictInput):
    object_ref: str | None = None
    tools: list[ToolCommandInput] = Field(default_factory=list, max_length=2)
    preserve_tool_state: bool = True
    object_may_be_released: bool | None = None
    reason: str
    mode: Literal["safe", "immediate"] | None = None


class RobotStateInput(StrictInput):
    pass


class VerifyToolLoadInput(StrictInput):
    tool_refs: list[str] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_tool_refs(self) -> "VerifyToolLoadInput":
        if len(self.tool_refs) != len(set(self.tool_refs)):
            raise ValueError("VerifyToolLoad 不能重复声明工具")
        return self


class CaptureRGBDInput(StrictInput):
    sensor_ids: list[str] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_sensors(self) -> "CaptureRGBDInput":
        if len(self.sensor_ids) != len(set(self.sensor_ids)):
            raise ValueError("CaptureRGBD 不能重复声明传感器")
        return self


class LocateObjectInput(StrictInput):
    object_ref: str
    approximate_region_ref: str | None = None
    pose_hint: PoseInput | None = None
    extent_hint_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    category_hint: str | None = None
    minimum_confidence: float = Field(ge=0.0, le=1.0)
    model_profile: str | None = None


class VerifyPregraspInput(StrictInput):
    object_ref: str
    candidate_id: str
    planned_object_pose: PoseInput
    expected_targets: list[EndEffectorTargetInput] = Field(min_length=1, max_length=2)
    target_revision: str
    maximum_position_error_m: float = Field(gt=0)
    model_profile: str | None = None

    @model_validator(mode="after")
    def validate_tools(self) -> "VerifyPregraspInput":
        _unique_tool_refs(self.expected_targets)
        return self


class VerifyGraspInput(StrictInput):
    object_ref: str
    tools: list[str] = Field(min_length=1, max_length=2)
    candidate_id: str
    initial_object_pose: PoseInput
    minimum_lift_height_m: float = Field(gt=0)
    lift_height_tolerance_m: float = Field(default=0.001, ge=0, le=0.01)
    stable_duration_ms: int = Field(ge=100, le=5000)
    model_profile: str | None = None


class ObservePlacementTargetInput(StrictInput):
    target_ref: str
    object_ref: str
    pose_hint: PoseInput | None = None
    extent_hint_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    category_hint: str | None = None
    require_free: bool = True
    require_reachable: bool = True
    model_profile: str | None = None


class VerifyPlacementInput(StrictInput):
    object_ref: str
    target_ref: str
    target_revision: str
    target_pose_hint: PoseInput | None = None
    target_extent_hint_m: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    stability_duration_ms: int = Field(gt=0)
    require_independent_source: bool = True
    model_profile: str | None = None


class GenerateCandidatesInput(StrictInput):
    object_ref: str
    target_pose: PoseInput
    object_extent_m: list[float] = Field(min_length=3, max_length=3)
    target_revision: str
    preferred_strategy: Literal[
        "auto", "direct_bilateral", "left_extract_first", "right_extract_first"
    ]
    maximum_candidates: int = Field(default=4, ge=1, le=12)
    engaged_tool_ref: str | None = None
    secondary_resume_phase: Literal["insert"] | None = None
    model_profile: str | None = None


class PlanTransportPostureInput(StrictInput):
    object_ref: str
    object_pose: PoseInput
    object_extent_m: list[float] = Field(min_length=3, max_length=3)
    target_revision: str
    tool_refs: list[str] = Field(min_length=2, max_length=2)
    model_profile: str | None = None

    @model_validator(mode="after")
    def validate_tool_refs(self) -> "PlanTransportPostureInput":
        if len(self.tool_refs) != len(set(self.tool_refs)):
            raise ValueError("PlanTransportPosture 不能重复声明工具")
        return self


class GetExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    invocation_id: str
    after_feedback_sequence: int = Field(default=0, ge=0)


class StopExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    invocation_id: str
    reason: str = "pilot stop"


TASK_INPUTS: dict[str, type[BaseModel]] = {
    "PlanRoute": PlanRouteInput,
    "FollowRoute": FollowRouteInput,
    "VerifyArrival": VerifyArrivalInput,
    "MoveEndEffector": MoveEndEffectorInput,
    "FollowWaypoints": FollowWaypointsInput,
    "LiftHeldObject": LiftHeldObjectInput,
    "MoveToPosture": MoveToPostureInput,
    "SetOpening": SetOpeningInput,
    "CloseUntilContact": CloseUntilContactInput,
    "Release": ReleaseInput,
    "HoldObject": HoldObjectInput,
    "GetRobotState": RobotStateInput,
    "VerifyToolLoad": VerifyToolLoadInput,
    "CaptureRGBD": CaptureRGBDInput,
    "LocateObject": LocateObjectInput,
    "VerifyPregrasp": VerifyPregraspInput,
    "VerifyGrasp": VerifyGraspInput,
    "ObservePlacementTarget": ObservePlacementTargetInput,
    "VerifyPlacement": VerifyPlacementInput,
    "GenerateCandidates": GenerateCandidatesInput,
    "PlanTransportPosture": PlanTransportPostureInput,
    "GetExecution": GetExecutionInput,
    "StopExecution": StopExecutionInput,
}
