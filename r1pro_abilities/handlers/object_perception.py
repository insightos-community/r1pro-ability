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

"""ObjectPerception Ability：Fake 与 MuJoCo 场景真值感知实现。"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence, TypeVar

from semantic_robot_sdk_core import Pose, SceneObject, SceneRegion, SceneSnapshot

from ..ability_utils import component_name, fixture_pose, observation, pose_dict
from ..execution import ExecutionRecord
from ..tool_load import (
    DEFAULT_TOOL_LOAD_POLICY,
    tools_are_unloaded,
    wait_for_stable_tool_load,
)
from .base import AbilityHandler

T = TypeVar("T", SceneObject, SceneRegion)


def _lift_height_meets_requirement(
    actual_m: float, minimum_m: float, tolerance_m: float
) -> bool:
    """按观测精度比较真实抬升高度，而不是要求浮点值精确命中命令值。

    这里的容差属于位姿观测契约，与 MuJoCo timestep、物理帧或箱型无关。
    双侧接触、受力、滑移和传感故障仍由独立实时状态验证；容差只用于
    比较物体前后 Pose，不能把没有真实抬升的箱体判成抓取成功。
    """

    return actual_m + tolerance_m >= minimum_m


class ObjectPerceptionHandler(AbilityHandler):
    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        if self.models is None:
            raise RuntimeError("ObjectPerception 缺少模型配置")
        profile = self.models.resolve("object_perception", data.get("model_profile"))
        if profile.provider == "fake":
            result, observations = self._fake(task_name, data, digest, profile.name)
        elif profile.provider == "mujoco_ground_truth":
            result, observations = self._mujoco(task_name, data, profile.name)
        else:
            raise RuntimeError(f"模型 Provider 尚未接入: {profile.provider}")
        return self.succeeded(
            task_name,
            data,
            invocation_id,
            digest,
            {"model_profile": profile.name, **result},
            observations,
        )

    def _mujoco(
        self,
        task_name: str,
        data: Mapping[str, Any],
        profile_name: str,
    ) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
        first = self.sdk.state.scene_snapshot()
        revision = _scene_revision(first)
        source = f"mujoco-ground-truth://{first.instance_id}"

        if task_name == "LocateObject":
            item = _resolve(
                first.objects,
                str(data["object_ref"]),
                pose_hint=data.get("pose_hint"),
                extent_hint=data.get("extent_hint_m"),
                category_hint=data.get("category_hint"),
            )
            pose = _pose(item.pose, first, revision)
            value = {
                "object_ref": data["object_ref"],
                "scene_source_id": item.source_id,
                "pose": pose,
                "extent_m": list(item.extent) if item.extent else None,
                "category": item.category,
                "scene_state": item.state,
                "identity_confidence": 1.0,
            }
            kind, subject, confidence = "target_pose", str(data["object_ref"]), 1.0
        elif task_name == "VerifyPregrasp":
            state = self.sdk.state.snapshot()
            item = _resolve(first.objects, str(data["object_ref"]))
            planned_object_pose = data["planned_object_pose"]
            planned_object_position = tuple(
                float(value) for value in planned_object_pose["position_m"]
            )
            planned_object_orientation = tuple(
                float(value) for value in planned_object_pose["orientation_xyzw"]
            )
            errors: dict[str, float] = {}
            object_relative_errors: dict[str, float] = {}
            engagement_errors: dict[str, float] = {}
            hook_contacts: dict[str, bool] = {}
            for target in data["expected_targets"]:
                tool_ref = str(target["tool_ref"])
                side = component_name(self.sdk, tool_ref, "object_perception")
                actual = state.end_effectors.get(side)
                if actual is None:
                    raise ValueError(
                        f"Robot SDK 未返回 {target['tool_ref']} 的末端位姿"
                    )
                expected_position = tuple(
                    float(value) for value in target["target_pose"]["position_m"]
                )
                errors[tool_ref] = _position_error(
                    actual.position, expected_position
                )
                expected_relative = _position_in_object_frame(
                    expected_position,
                    planned_object_position,
                    planned_object_orientation,
                )
                actual_relative = _position_in_object_frame(
                    actual.position,
                    item.pose.position,
                    item.pose.quaternion_xyzw,
                )
                object_relative_errors[tool_ref] = _position_error(
                    actual_relative, expected_relative
                )
                # 周转箱短边接口是一条沿物体局部 Y 方向延伸的凹槽，而不是
                # 一个必须命中单点的“承力条”。插入阶段真正决定钩脚能否
                # 进入槽内的是局部 X 深度和 Z 高度；箱体被首侧承载时产生的
                # 少量槽长方向漂移，不应在 seat 之前被标成抓取失败。最终是否
                # 钩入仍由 seat 后的真实 hook contact/force 证明。
                engagement_errors[tool_ref] = _groove_engagement_error(
                    actual_relative, expected_relative
                )
                tool_state = state.tool_states.get(tool_ref)
                hook_contacts[tool_ref] = bool(tool_state and tool_state.hook_contact)
            # 末端到达世界坐标并不等于下钩已经进入凹槽：钩脚撞在槽口时会
            # 把自由箱体一起推走，世界坐标误差仍可能接近零。这里同时验证
            # 工具相对实时箱体的插入关系；Runtime只提供原始Pose和接触，
            # “是否进入本次候选的凹槽”仍由感知Ability判断。
            maximum_error = float(data["maximum_position_error_m"])
            reached = (
                bool(errors)
                and max(errors.values()) <= maximum_error
                and max(engagement_errors.values()) <= maximum_error
                and all(
                    state.tool_states.get(str(target["tool_ref"])) is not None
                    and not state.tool_states[str(target["tool_ref"])].sensor_fault
                    for target in data["expected_targets"]
                )
            )
            value = {
                "candidate_id": data["candidate_id"],
                "target_revision": data["target_revision"],
                "reached": reached,
                "position_errors_m": errors,
                "object_relative_errors_m": object_relative_errors,
                "engagement_errors_m": engagement_errors,
                "hook_contacts": hook_contacts,
                "scene_revision": revision,
            }
            kind, subject, confidence = "pregrasp_state", str(data["object_ref"]), 1.0
        elif task_name == "VerifyGrasp":
            stable_duration = int(data["stable_duration_ms"]) / 1000.0
            policy = replace(
                DEFAULT_TOOL_LOAD_POLICY,
                stable_duration_s=stable_duration,
                acquisition_timeout_s=max(
                    DEFAULT_TOOL_LOAD_POLICY.acquisition_timeout_s,
                    stable_duration + 0.5,
                ),
            )
            state_after, load, observed_duration_ms = wait_for_stable_tool_load(
                self.sdk, [str(ref) for ref in data["tools"]], policy
            )
            scene_after = self.sdk.state.scene_snapshot()
            item = _resolve(
                scene_after.objects,
                str(data["object_ref"]),
                pose_hint=data.get("initial_object_pose"),
            )
            expected = str(data["object_ref"])
            tool_states = load.tool_states
            stable_tools = load.safe
            lift_height = item.pose.position[2] - float(
                data["initial_object_pose"]["position_m"][2]
            )
            held = stable_tools and _lift_height_meets_requirement(
                lift_height,
                float(data["minimum_lift_height_m"]),
                float(data.get("lift_height_tolerance_m", 0.001)),
            )
            value = {
                "candidate_id": data["candidate_id"],
                "held": held,
                "stable_bilateral_load": stable_tools,
                "lift_height_m": lift_height,
                "lift_height_tolerance_m": float(
                    data.get("lift_height_tolerance_m", 0.001)
                ),
                "stable_duration_ms": observed_duration_ms,
                "slipping": load.slipping,
                "overloaded": load.overloaded,
                "sensor_fault": load.sensor_fault,
                "reasons": list(load.reasons),
                "object_pose": _pose(
                    item.pose, scene_after, _scene_revision(scene_after)
                ),
                "tool_poses": {
                    ref: pose_dict(
                        state_after.end_effectors[
                            component_name(self.sdk, str(ref), "object_perception")
                        ],
                        revision=_scene_revision(scene_after),
                    )
                    for ref in data["tools"]
                },
                "tool_states": {
                    key: item.model_dump(mode="json") if item else None
                    for key, item in tool_states.items()
                },
            }
            revision = _scene_revision(scene_after)
            kind, subject, confidence = "grasp_verification", expected, 1.0
        elif task_name == "ObservePlacementTarget":
            region = _resolve(
                first.regions,
                str(data["target_ref"]),
                pose_hint=data.get("pose_hint"),
                extent_hint=data.get("extent_hint_m"),
            )
            occupants = [
                item
                for item in first.objects
                if item.source_id != str(data["object_ref"])
                and _within_region(item, region)
            ]
            moving_object = _resolve(first.objects, str(data["object_ref"]))
            placement_pose, free = _next_column_pose(
                region,
                occupants,
                moving_object,
                first,
                revision,
            )
            support_surface_ref = str(
                region.properties.get("support_surface_ref") or data["target_ref"]
            )
            support_surface = next(
                (
                    item for item in first.objects
                    if item.source_id == support_surface_ref
                ),
                None,
            )
            support_occupied = bool(
                support_surface is not None
                and support_surface.extent is not None
                and any(
                    item.source_id not in {
                        moving_object.source_id,
                        support_surface.source_id,
                    }
                    and item.category == moving_object.category
                    and abs(item.pose.position[0] - support_surface.pose.position[0])
                    <= support_surface.extent[0] / 2
                    and abs(item.pose.position[1] - support_surface.pose.position[1])
                    <= support_surface.extent[1] / 2
                    for item in first.objects
                )
            )
            lateral_clearance_m = _placement_lateral_clearances(
                first, moving_object, placement_pose
            )
            value = {
                "schema_version": 2,
                "target_ref": data["target_ref"],
                "region_ref": data["target_ref"],
                "placement_pose": placement_pose,
                "approach_vector": [0.0, 0.0, 1.0],
                "extent_m": list(region.extent) if region.extent else None,
                "free": free,
                "reachable": True,
                "occupants": [item.source_id for item in occupants],
                "support_surface_ref": support_surface_ref,
                "support_center_pose": (
                    _pose(support_surface.pose, first, revision)
                    if support_surface is not None else None
                ),
                "support_occupied": support_occupied,
                "lateral_clearance_m": lateral_clearance_m,
                "revision": revision,
                "confidence": 1.0,
                "evidence_refs": [],
            }
            kind, subject, confidence = (
                "placement.target_slot",
                str(data["target_ref"]),
                1.0,
            )
        elif task_name == "VerifyPlacement":
            region = _resolve(
                first.regions,
                str(data["target_ref"]),
                pose_hint=data.get("target_pose_hint"),
                extent_hint=data.get("target_extent_hint_m"),
            )
            before = _resolve(
                first.objects,
                str(data["object_ref"]),
                pose_hint=data.get("target_pose_hint"),
            )
            time.sleep(int(data["stability_duration_ms"]) / 1000.0)
            second = self.sdk.state.scene_snapshot()
            after = _resolve(
                second.objects,
                str(data["object_ref"]),
                pose_hint=data.get("target_pose_hint"),
            )
            robot = self.sdk.state.snapshot()
            displacement = _position_error(before.pose.position, after.pose.position)
            within = _within_region(after, region)
            descriptors = self.sdk.state.capabilities().tools
            gripper_empty = tools_are_unloaded(
                robot, descriptors, list(robot.tool_states)
            )
            stable = (
                within
                and displacement <= 0.01
                and bool(after.state.get("in_contact"))
                and gripper_empty
            )
            revision = _scene_revision(second)
            final_pose = _pose(after.pose, second, revision)
            target_pose_hint = data.get("target_pose_hint")
            expected_position = (
                tuple(float(value) for value in target_pose_hint["position_m"])
                if isinstance(target_pose_hint, Mapping)
                else region.pose.position
            )
            # Region 的 Z 表达托盘支撑面，实时规划得到的 placement pose 才表达
            # 箱体中心。二者天然相差半个箱高，不能把这个几何语义差异误判成
            # 放置偏差。XY 是否落在目标列、是否真实接触支撑面仍独立校验。
            position_error = _position_error(after.pose.position, expected_position)
            orientation_error = _orientation_error(
                after.pose.quaternion_xyzw,
                region.pose.quaternion_xyzw,
            )
            placed_state = {
                "object_ref": data["object_ref"],
                "target_ref": data["target_ref"],
                "final_pose": final_pose,
                "support_surface_ref": str(
                    region.properties.get("support_surface_ref") or region.source_id
                ),
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "stable": stable,
                "gripper_empty": gripper_empty,
                "verification_source": source,
                "observed_duration_ms": data["stability_duration_ms"],
                "scene_revision": revision,
                "evidence_refs": [],
            }
            value = {
                "state": placed_state,
                "within_target": within,
                "stable": stable,
                "observed_displacement_m": displacement,
                "observed_duration_ms": data["stability_duration_ms"],
                "support_contact": bool(after.state.get("in_contact")),
                "gripper_empty": gripper_empty,
                "independent_verification": True,
            }
            kind, subject, confidence = (
                "placement.object_stability",
                str(data["object_ref"]),
                1.0,
            )
        else:
            raise ValueError(f"ObjectPerception 不支持 Task: {task_name}")

        obs = observation(
            kind,
            source,
            subject,
            value,
            revision=revision,
            frame_id=first.coordinate_frame,
            confidence=confidence,
        )
        return {"verification": value}, (obs,)

    def _fake(
        self,
        task_name: str,
        data: Mapping[str, Any],
        digest: str,
        profile_name: str,
    ) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
        if task_name == "LocateObject":
            revision = f"target-{digest[:12]}"
            target_pose = fixture_pose(revision)
            value = {
                "object_ref": data["object_ref"],
                "pose": target_pose,
                "extent_m": [0.60, 0.40, 0.34],
                "identity_confidence": 0.95,
            }
            kind, subject = "target_pose", str(data["object_ref"])
        elif task_name == "VerifyPregrasp":
            revision = str(data["target_revision"])
            value = {
                "candidate_id": data["candidate_id"],
                "target_revision": data["target_revision"],
                "reached": True,
                "position_errors_m": {
                    item["tool_ref"]: 0.005 for item in data["expected_targets"]
                },
                "object_relative_errors_m": {
                    item["tool_ref"]: 0.0 for item in data["expected_targets"]
                },
                "hook_contacts": {
                    item["tool_ref"]: True for item in data["expected_targets"]
                },
            }
            kind, subject = "pregrasp_state", str(data["object_ref"])
        elif task_name == "VerifyGrasp":
            state = self.sdk.state.snapshot()
            revision = f"robot-state-{digest[:12]}"
            stable_duration = int(data["stable_duration_ms"]) / 1000.0
            policy = replace(
                DEFAULT_TOOL_LOAD_POLICY,
                stable_duration_s=stable_duration,
                acquisition_timeout_s=max(
                    DEFAULT_TOOL_LOAD_POLICY.acquisition_timeout_s,
                    stable_duration + 0.5,
                ),
            )
            state, load, observed_duration_ms = wait_for_stable_tool_load(
                self.sdk, [str(ref) for ref in data["tools"]], policy
            )
            held = load.safe
            value = {
                "candidate_id": data["candidate_id"],
                "held": held,
                "stable_bilateral_load": held,
                "lift_height_m": data["minimum_lift_height_m"],
                "stable_duration_ms": observed_duration_ms,
                "slipping": load.slipping,
                "overloaded": load.overloaded,
                "sensor_fault": load.sensor_fault,
                "reasons": list(load.reasons),
                "object_pose": fixture_pose(revision),
                "tool_poses": {
                    ref: pose_dict(
                        state.end_effectors[
                            component_name(self.sdk, str(ref), "object_perception")
                        ],
                        revision=revision,
                    )
                    for ref in data["tools"]
                },
            }
            kind, subject = "grasp_verification", str(data["object_ref"])
        elif task_name == "ObservePlacementTarget":
            revision = f"slot-{digest[:12]}"
            value = {
                "schema_version": 2,
                "target_ref": data["target_ref"],
                "region_ref": data["target_ref"],
                "placement_pose": fixture_pose(revision, x=0.8, z=0.75),
                "approach_vector": [0.0, 0.0, 1.0],
                "free": True,
                "reachable": True,
                "support_surface_ref": data["target_ref"],
                "revision": revision,
                "confidence": 1.0,
                "evidence_refs": [],
            }
            kind, subject = "placement.target_slot", str(data["target_ref"])
        elif task_name == "VerifyPlacement":
            revision = str(data["target_revision"])
            robot = self.sdk.state.snapshot()
            empty = all(
                not tool.hook_contact and not tool.clamp_contact
                for tool in robot.tool_states.values()
            )
            final_pose = fixture_pose(revision, x=0.8, z=0.75)
            source = profile_name
            state = {
                "object_ref": data["object_ref"],
                "target_ref": data["target_ref"],
                "final_pose": final_pose,
                "support_surface_ref": data["target_ref"],
                "position_error_m": 0.0,
                "orientation_error_rad": 0.0,
                "stable": empty,
                "gripper_empty": empty,
                "verification_source": source,
                "observed_duration_ms": data["stability_duration_ms"],
                "scene_revision": revision,
                "evidence_refs": [],
            }
            value = {
                "state": state,
                "within_target": True,
                "stable": empty,
                "observed_displacement_m": 0.0,
                "observed_duration_ms": data["stability_duration_ms"],
                "support_contact": True,
                "gripper_empty": empty,
                "independent_verification": True,
            }
            kind, subject = "placement.object_stability", str(data["object_ref"])
        else:
            raise ValueError(f"ObjectPerception 不支持 Task: {task_name}")
        obs = observation(
            kind, profile_name, subject, value, revision=revision, confidence=1.0
        )
        return {"verification": value}, (obs,)


def _scene_revision(snapshot: SceneSnapshot) -> str:
    return f"scene:{snapshot.instance_id}:{snapshot.generation}:{snapshot.observed_at.isoformat()}"


def _resolve(
    items: Sequence[T],
    reference: str,
    *,
    pose_hint: Mapping[str, Any] | None = None,
    extent_hint: Sequence[float] | None = None,
    category_hint: str | None = None,
) -> T:
    """在当前场景快照中解析目标，提示只用于关联候选。

    精确 Runtime source ID 最可靠。Agent 传入 Map entity ID 时，Ability 可以
    使用随任务携带的位姿、尺寸和类别缩小当前快照中的候选；后续控制只使用
    这里重新观测到的实时 Pose。这里不设置地图误差阈值，因为 Semantic Map
    是规划记忆，不是 Robot Skill 的执行事实。
    """

    exact = [item for item in items if item.source_id == reference]
    if len(exact) == 1:
        return exact[0]
    leaf = reference.rstrip("/").rsplit("/", 1)[-1]
    aliases = [item for item in items if item.source_id == leaf]
    if len(aliases) == 1:
        return aliases[0]

    candidates = list(items)
    if category_hint:
        matching = [
            item
            for item in candidates
            if getattr(item, "category", None) == category_hint
        ]
        # 类别是提示；旧地图类别与当前 Runtime 不一致时不应直接清空候选。
        if matching:
            candidates = matching
    if len(candidates) == 1:
        return candidates[0]

    pose_position = (pose_hint or {}).get("position_m")
    can_score_pose = isinstance(pose_position, Sequence) and len(pose_position) == 3
    can_score_extent = extent_hint is not None and len(extent_hint) == 3
    if candidates and (can_score_pose or can_score_extent):
        scored: list[tuple[float, T]] = []
        for item in candidates:
            score = 0.0
            if can_score_pose:
                score += _position_error(item.pose.position, pose_position)
            if can_score_extent and item.extent is not None:
                score += _position_error(item.extent, extent_hint)
            scored.append((score, item))
        scored.sort(key=lambda value: value[0])
        # 分数完全相同时无法诚实判断是哪一个现场对象，交给上层澄清。
        if len(scored) == 1 or not math.isclose(
            scored[0][0], scored[1][0], rel_tol=1e-9, abs_tol=1e-9
        ):
            return scored[0][1]
    raise ValueError(f"场景快照中无法唯一解析引用: {reference}")


def _pose(value: Pose, snapshot: SceneSnapshot, revision: str) -> dict[str, Any]:
    return {
        "frame_id": value.frame_id,
        "position_m": list(value.position),
        "orientation_xyzw": list(value.quaternion_xyzw),
        "observed_at": snapshot.observed_at.isoformat(),
        "revision": revision,
    }


def _position_in_object_frame(
    world_position: Sequence[float],
    object_position: Sequence[float],
    object_orientation_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    """把世界坐标点转换成物体局部坐标，供接合关系复核。"""

    delta = tuple(
        float(world_position[index]) - float(object_position[index])
        for index in range(3)
    )
    x, y, z, w = (float(value) for value in object_orientation_xyzw)
    # 使用物体四元数的共轭旋转 world -> object。这里只处理实时几何，
    # 不读取Semantic Map，也不让Runtime理解周转箱抓取语义。
    ux, uy, uz = -x, -y, -z
    dot_uv = ux * delta[0] + uy * delta[1] + uz * delta[2]
    dot_uu = ux * ux + uy * uy + uz * uz
    cross = (
        uy * delta[2] - uz * delta[1],
        uz * delta[0] - ux * delta[2],
        ux * delta[1] - uy * delta[0],
    )
    return tuple(
        2.0 * dot_uv * (ux, uy, uz)[index]
        + (w * w - dot_uu) * delta[index]
        + 2.0 * w * cross[index]
        for index in range(3)
    )


def _groove_engagement_error(
    actual_relative: Sequence[float], expected_relative: Sequence[float]
) -> float:
    """返回短边凹槽插入深度与高度的合成误差。

    凹槽沿物体局部 Y 方向延伸；Y 向偏移只改变钩脚在同一槽内的位置，
    不改变是否已经水平插入。最终钩入仍必须由 seat 后的真实接触证明。
    """

    return math.hypot(
        float(actual_relative[0]) - float(expected_relative[0]),
        float(actual_relative[2]) - float(expected_relative[2]),
    )


def _position_error(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))
    )


def _orientation_error(left: Sequence[float], right: Sequence[float]) -> float:
    """返回两个单位四元数之间的最短旋转角。"""

    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _within_region(item: SceneObject, region: SceneRegion) -> bool:
    if region.extent is None:
        return _position_error(item.pose.position[:2], region.pose.position[:2]) <= 0.05
    return all(
        abs(item.pose.position[index] - region.pose.position[index])
        <= region.extent[index] / 2
        for index in (0, 1)
    )


def _placement_lateral_clearances(
    snapshot: SceneSnapshot,
    moving_object: SceneObject,
    placement_pose: Mapping[str, Any],
) -> dict[str, float | None]:
    """返回最终放置高度上，箱体局部X两侧到相邻物体的真实净空。

    这只是ObjectPerception基于当前SceneSnapshot形成的动作观测，不是
    Semantic Map实体，也不让Runtime理解码垛策略。Skill用它选择先撤哪一侧
    工具；实际轨迹仍由SDK逐点做碰撞检查。
    """

    if moving_object.extent is None:
        return {"negative": None, "positive": None}
    position = tuple(float(value) for value in placement_pose["position_m"])
    orientation = tuple(
        float(value) for value in placement_pose["orientation_xyzw"]
    )
    target_axes = tuple(
        _rotate_vector(orientation, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    moving_half = tuple(float(value) / 2.0 for value in moving_object.extent)
    clearances: dict[str, float | None] = {"negative": None, "positive": None}
    for item in snapshot.objects:
        if (
            item.source_id == moving_object.source_id
            or item.category != moving_object.category
            or item.extent is None
        ):
            continue
        delta = tuple(
            float(item.pose.position[index]) - position[index]
            for index in range(3)
        )
        local_center = tuple(
            sum(delta[index] * axis[index] for index in range(3))
            for axis in target_axes
        )
        item_axes = tuple(
            _rotate_vector(item.pose.quaternion_xyzw, axis)
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        projected_half = tuple(
            sum(
                abs(sum(target_axis[index] * item_axis[index] for index in range(3)))
                * float(item.extent[item_index])
                / 2.0
                for item_index, item_axis in enumerate(item_axes)
            )
            for target_axis in target_axes
        )
        # 只把与目标箱体同一高度、同一行真实重叠的物体看作侧向邻箱；
        # 下层支撑箱和另一行箱体不能错误触发逐侧退出策略。
        if (
            moving_half[1] + projected_half[1] - abs(local_center[1]) <= 0.002
            or moving_half[2] + projected_half[2] - abs(local_center[2]) <= 0.002
        ):
            continue
        side = "positive" if local_center[0] >= 0.0 else "negative"
        clearance = max(
            0.0,
            abs(local_center[0]) - moving_half[0] - projected_half[0],
        )
        current = clearances[side]
        clearances[side] = clearance if current is None else min(current, clearance)
    return clearances


def _rotate_vector(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    """使用单位四元数把局部向量旋转到世界坐标。"""

    x, y, z, w = (float(value) for value in quaternion_xyzw)
    vx, vy, vz = (float(value) for value in vector)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("场景对象姿态不是有效四元数")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    dot_uv = x * vx + y * vy + z * vz
    dot_uu = x * x + y * y + z * z
    cross = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    return tuple(
        2.0 * dot_uv * (x, y, z)[index]
        + (w * w - dot_uu) * (vx, vy, vz)[index]
        + 2.0 * w * cross[index]
        for index in range(3)
    )


def _next_column_pose(
    region: SceneRegion,
    occupants: Sequence[SceneObject],
    moving_object: SceneObject,
    snapshot: SceneSnapshot,
    revision: str,
) -> tuple[dict[str, Any], bool]:
    """从普通 Region 属性和实时几何推导下一箱体中心位姿。

    Runtime 只负责返回当前对象与 Region；“第几层、是否兼容、下一层多高”是
    ObjectPerception 对本次放置动作的实时判断。这里使用每个物体中心 Pose 与
    AABB extent，不使用 Semantic Map 的历史高度，也不引入 PlacementPattern。
    """

    if moving_object.extent is None:
        raise ValueError("待放物体缺少实时 extent，无法计算堆叠高度")
    properties = region.properties
    max_layers = int(properties.get("max_layers", 1))
    accepted = {str(item) for item in properties.get("accepted_models", [])}
    model = str(moving_object.state.get("model") or "")
    occupant_models = {str(item.state.get("model") or "") for item in occupants}
    compatible = (not accepted or model in accepted) and all(
        not accepted or current in accepted for current in occupant_models
    )
    if any(item.extent is None for item in occupants):
        raise ValueError("堆叠列已有物体缺少实时 extent")

    support_z = float(
        properties.get(
            "support_z_m",
            region.pose.position[2] - (region.extent[2] / 2 if region.extent else 0.0),
        )
    )
    top_z = max(
        [support_z]
        + [
            item.pose.position[2] + item.extent[2] / 2
            for item in occupants
            if item.extent
        ]
    )
    center_z = top_z + moving_object.extent[2] / 2
    pose = Pose(
        position=(region.pose.position[0], region.pose.position[1], center_z),
        quaternion_xyzw=region.pose.quaternion_xyzw,
        frame_id=region.pose.frame_id,
    )
    return _pose(pose, snapshot, revision), compatible and len(occupants) < max_layers
