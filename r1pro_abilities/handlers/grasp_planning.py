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

"""GraspPlanning Ability：生成周转箱双侧夹具的三种语义候选。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from semantic_robot_sdk_core import (
    EnvironmentCollisionObject,
    EnvironmentCollisionSet,
    PlanningError,
    Pose,
)

from ..ability_utils import observation, scene_collision_environment
from ..execution import ExecutionRecord
from .base import AbilityHandler

STRATEGIES = ("direct_bilateral", "left_extract_first", "right_extract_first")


class GraspPlanningHandler(AbilityHandler):
    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        if task_name not in {"GenerateCandidates", "PlanTransportPosture"}:
            raise ValueError(f"GraspPlanning 不支持 Task: {task_name}")
        if self.models is None:
            raise RuntimeError("GraspPlanning 缺少模型配置")
        profile = self.models.resolve("grasp_planning", data.get("model_profile"))
        if profile.provider not in {"fake", "mujoco_ground_truth"}:
            raise RuntimeError(f"模型 Provider 尚未接入: {profile.provider}")
        if task_name == "PlanTransportPosture":
            return self._plan_transport_posture(
                data, invocation_id, digest, profile.settings, profile.name
            )

        tools = self.sdk.state.capabilities().tools
        by_side = {tool.side: tool for tool in tools if tool.kind == "tote_clamp"}
        if set(by_side) != {"left", "right"}:
            raise ValueError("周转箱候选要求 Robot Profile 精确声明左右 tote_clamp")
        preferred = str(data["preferred_strategy"])
        robot_state = self.sdk.state.snapshot()
        if robot_state.base_pose is None:
            raise RuntimeError("抓取规划需要 Robot 实时底盘位姿")
        ordered = list(STRATEGIES)
        scene_objects: Sequence[Any] = ()
        if profile.provider == "mujoco_ground_truth":
            try:
                scene = self.sdk.state.scene_snapshot()
            except RuntimeError as exc:
                # 真机或单元测试Backend可以没有场景接口；目标Pose和extent已经
                # 是本次规划的充分输入。只有明确“不提供接口”时退化为无邻箱
                # 候选，真实MuJoCo Runtime的连接或读取错误仍须原样失败。
                if str(exc) != "当前 Robot Backend 不提供 SceneSnapshot":
                    raise
            else:
                scene_objects = scene.objects
                ordered = _order_strategies_for_scene(
                    ordered, data, profile.settings, robot_state.base_pose, scene_objects
                )
        engaged_tool_ref = data.get("engaged_tool_ref")
        if engaged_tool_ref:
            engaged_side = next(
                (
                    side
                    for side, tool in by_side.items()
                    if tool.tool_ref == str(engaged_tool_ref)
                ),
                None,
            )
            if engaged_side is None:
                raise ValueError("已接合工具不属于当前Robot")
            # 一侧已经形成真实接触后，抓取策略不再是三个独立候选：首侧身份
            # 已由物理现场确定，direct与反向extract都会重复计算同一条第二侧
            # 进入路径，反向策略更不可能在不释放现场的前提下执行。这里只保留
            # 与已接合侧一致的候选；路径已可达时下方会把它收敛为direct。
            ordered = [f"{engaged_side}_extract_first"]
        else:
            if preferred != "auto":
                ordered.remove(preferred)
                ordered.insert(0, preferred)
            ordered = ordered[: min(int(data["maximum_candidates"]), len(ordered))]
        motion_position_tolerance = max(
            1e-4,
            float(
                getattr(
                    self.sdk.kinematics_provider,
                    "fixed_position_tolerance_m",
                    1e-4,
                )
            ),
        )
        candidate_requests = [
            (strategy, index, data)
            for index, strategy in enumerate(ordered, start=1)
        ]
        if engaged_tool_ref and data.get("secondary_resume_phase") == "insert":
            # 第二只手已到槽口外，此时不是重新选择抓取策略，而是在同一条
            # 连续凹槽内选择邻近接入点。最多返回调用方已请求的三个候选；
            # Skill只会在前一个点规划失败且没有物理副作用时尝试下一个。
            candidate_requests = [
                (
                    ordered[0],
                    index,
                    {**data, "_secondary_lane_candidate_index": index - 1},
                )
                for index in range(1, min(int(data["maximum_candidates"]), 3) + 1)
            ]
        geometric_candidates = [
            _candidate(
                strategy,
                index,
                candidate_data,
                by_side,
                profile.settings,
                robot_state,
                scene_objects,
                motion_position_tolerance_m=motion_position_tolerance,
            )
            for strategy, index, candidate_data in candidate_requests
        ]
        candidates: list[dict[str, Any]] = []
        rejected: list[str] = []
        for candidate in geometric_candidates:
            reachable, reason = _candidate_reachability(
                self.sdk,
                candidate,
                robot_state,
                by_side,
                engaged_tool_ref=data.get("engaged_tool_ref"),
                object_ref=str(data["object_ref"]),
                secondary_resume_phase=data.get("secondary_resume_phase"),
            )
            if reachable:
                deferred_reason = candidate.pop("_deferred_secondary_reason", None)
                candidate.pop("_secondary_obstacle_ids", None)
                if deferred_reason:
                    rejected.append(str(deferred_reason))
                candidate.pop("_secondary_preflight_mode", None)
                candidate.pop("_secondary_resume_tolerance_m", None)
                candidate.pop("_target_collision_geometry", None)
                if (
                    engaged_tool_ref
                    and candidate["strategy"] != "direct_bilateral"
                    and not candidate.get("pull_path")
                ):
                    candidate["strategy"] = "direct_bilateral"
                    candidate["candidate_id"] = (
                        "direct_bilateral-"
                        + candidate["candidate_id"].rsplit("-", 1)[-1]
                    )
                candidates.append(candidate)
                # 外拉后的目标已经重新观测，direct 候选一旦可用就是下一步
                # 真正要执行的方案。继续把另外两种策略各自完整预规划一遍
                # 不会增加执行证据，只会让已接合工具在等待期间逐渐卸载。
                if engaged_tool_ref and data.get("secondary_resume_phase") != "insert":
                    break
            elif reason:
                rejected.append(reason)
        if not candidates:
            details = "；".join(rejected)
            message = "当前底盘站位下没有可达抓取候选，需要先调整接近位姿并重新观测"
            if details:
                message += f"（{details}）"
            return self.failed(
                task_name,
                data,
                invocation_id,
                digest,
                "GRASP_SITE_UNREACHABLE",
                message,
            )
        obs = observation(
            "grasp_candidates",
            profile.name,
            str(data["object_ref"]),
            {"target_revision": data["target_revision"], "candidates": candidates},
            revision=str(data["target_revision"]),
            confidence=1.0 if profile.provider == "mujoco_ground_truth" else 0.95,
        )
        return self.succeeded(
            task_name,
            data,
            invocation_id,
            digest,
            {
                "model_profile": profile.name,
                "candidates": candidates,
                # 被过滤候选的底层IK/碰撞原因只作为本次规划诊断返回，便于
                # Execution Inspector解释为何继续外拉；它不参与Skill控制，
                # 也不引入新的状态或校验条件。
                "rejected_candidates": rejected,
            },
            (obs,),
        )

    def _plan_transport_posture(
        self,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
        settings: Mapping[str, Any],
        profile_name: str,
    ) -> ExecutionRecord:
        """按真实抓取关系生成低扫掠携物姿态。

        抓取前的理想seat位姿不能代表夹紧后的工具—箱体关系。真实接触会让
        两侧产生不同的微小沉降；若把理想位姿平移成双臂目标，闭环机构会在
        运动终点互相顶住。这里在抬升验证后读取实时末端，只对箱体和两末端
        施加同一个平移，保留本次真实抓取形成的相对关系。
        """

        robot_state = self.sdk.state.snapshot()
        if robot_state.base_pose is None:
            raise RuntimeError("携物姿态规划需要Robot实时底盘位姿")
        descriptors = {
            tool.tool_ref: tool
            for tool in self.sdk.state.capabilities().tools
            if tool.kind == "tote_clamp"
        }
        requested = [str(value) for value in data["tool_refs"]]
        if set(requested) - set(descriptors):
            raise ValueError("携物姿态引用了当前Robot不存在的tote_clamp工具")
        sides = {descriptors[ref].side for ref in requested}
        if sides != {"left", "right"}:
            raise ValueError("携物姿态要求左右两侧tote_clamp")

        object_pose = data["object_pose"]
        object_center = tuple(float(value) for value in object_pose["position_m"])
        extent = tuple(float(value) for value in data["object_extent_m"])
        base_pose = robot_state.base_pose
        configured_offset = tuple(
            float(value)
            for value in settings.get(
                "transport_object_offset_base_m", [0.55, 0.0, 0.62]
            )
        )
        if len(configured_offset) != 3:
            raise ValueError("transport_object_offset_base_m 必须包含三个坐标")
        world_offset = _rotate_vector(base_pose.quaternion_xyzw, configured_offset)
        desired_center = [
            float(base_pose.position[index]) + world_offset[index] for index in range(3)
        ]
        minimum_bottom_clearance = float(
            settings.get("minimum_transport_bottom_clearance_m", 0.12)
        )
        desired_center[2] = max(
            desired_center[2],
            float(base_pose.position[2]) + extent[2] / 2.0 + minimum_bottom_clearance,
        )
        # prepare_transport发生在Robot尚未退出来源垛时。低层箱可以抬到最低
        # 行走高度，但顶层箱不能为了追求固定高度先向下穿过相邻箱；它必须先在
        # 当前真实高度向Robot中心收拢。后续放置下降仍由place-object根据目标
        # 列实时几何规划，不在这里写L1/L2/L3模板。
        desired_center[2] = max(desired_center[2], object_center[2])

        base_forward_world = _rotate_vector(
            base_pose.quaternion_xyzw, (1.0, 0.0, 0.0)
        )
        base_lateral_world = _rotate_vector(
            base_pose.quaternion_xyzw, (0.0, 1.0, 0.0)
        )
        base_origin = tuple(float(value) for value in base_pose.position)

        def base_projection(position: Sequence[float], axis: Sequence[float]) -> float:
            return _dot(
                tuple(float(position[index]) - base_origin[index] for index in range(3)),
                axis,
            )

        current_forward = base_projection(object_center, base_forward_world)
        current_lateral = base_projection(object_center, base_lateral_world)
        configured_forward = base_projection(desired_center, base_forward_world)
        desired_lateral = base_projection(desired_center, base_lateral_world)
        clearance_forward = configured_forward
        carried_half_forward = 0.5 * sum(
            abs(base_forward_world[index]) * extent[index] for index in range(3)
        )
        carried_half_lateral = 0.5 * sum(
            abs(base_lateral_world[index]) * extent[index] for index in range(3)
        )
        carried_half_height = extent[2] / 2.0
        path_margin = float(settings.get("tool_access_envelope_m", 0.0))

        # 目标箱已被抬起，但在退出同层邻箱的前后包络前仍不能横向居中。
        # SceneSnapshot提供的是实时几何事实；这里用它推导本次需要退到多近，
        # 不写死layout001坐标，也不把Semantic Map当作运动依据。
        scene = self.sdk.state.scene_snapshot()
        swept_lateral_min = min(current_lateral, desired_lateral) - carried_half_lateral
        swept_lateral_max = max(current_lateral, desired_lateral) + carried_half_lateral
        for item in scene.objects:
            if item.source_id == str(data["object_ref"]) or item.extent is None:
                continue
            other_extent = tuple(float(value) for value in item.extent)
            if len(other_extent) != 3:
                continue
            if (
                abs(float(item.pose.position[2]) - desired_center[2])
                >= carried_half_height + other_extent[2] / 2.0
            ):
                continue
            other_lateral = base_projection(item.pose.position, base_lateral_world)
            other_half_lateral = 0.5 * sum(
                abs(base_lateral_world[index]) * other_extent[index]
                for index in range(3)
            )
            if (
                other_lateral + other_half_lateral <= swept_lateral_min
                or other_lateral - other_half_lateral >= swept_lateral_max
            ):
                continue
            other_forward = base_projection(item.pose.position, base_forward_world)
            other_half_forward = 0.5 * sum(
                abs(base_forward_world[index]) * other_extent[index]
                for index in range(3)
            )
            safe_forward = (
                other_forward
                - other_half_forward
                - carried_half_forward
                - path_margin
            )
            if safe_forward < current_forward:
                clearance_forward = min(clearance_forward, safe_forward)

            # 退出邻箱包络以后还要恢复Profile定义的正常前向携物距离，
            # 否则箱体会永久贴在底盘前方，Robot Agent按工作距离规划出的
            # 单次导航目标就会失效。若邻箱在该前向距离仍与箱体重叠，只把
            # 最终携物中心向当前无障碍侧偏开所需的最小量；不强求绝对居中，
            # 也不重新把箱体推回来源垛。
            final_forward_overlap = (
                abs(other_forward - configured_forward)
                < other_half_forward + carried_half_forward + path_margin
            )
            if final_forward_overlap:
                if current_lateral >= other_lateral:
                    desired_lateral = max(
                        desired_lateral,
                        other_lateral
                        + other_half_lateral
                        + carried_half_lateral
                        + path_margin,
                    )
                else:
                    desired_lateral = min(
                        desired_lateral,
                        other_lateral
                        - other_half_lateral
                        - carried_half_lateral
                        - path_margin,
                    )

        configured_lateral = base_projection(desired_center, base_lateral_world)
        for index in range(3):
            desired_center[index] += (
                desired_lateral - configured_lateral
            ) * base_lateral_world[index]
        delta = tuple(
            desired_center[index] - object_center[index] for index in range(3)
        )

        # 密集堆垛中不能一边退出托盘一边横向居中，否则靠近相邻箱的一侧
        # 钩脚会在尚未获得前后净空时扫过邻箱。第一段保留当前横向抓取位置，
        # 只沿前后/竖直方向退出；第二段才进入可导航的正常携物姿态。
        vertical_delta = desired_center[2] - object_center[2]
        clearance_delta = tuple(
            (clearance_forward - current_forward) * base_forward_world[index]
            + (vertical_delta if index == 2 else 0.0)
            for index in range(3)
        )

        def translated_tool_poses(translation: Sequence[float]) -> list[dict[str, Any]]:
            values: list[dict[str, Any]] = []
            for ref in requested:
                side = descriptors[ref].side
                current = robot_state.end_effectors.get(side)
                if current is None:
                    raise RuntimeError(f"Robot状态缺少{side}末端位姿")
                if str(current.frame_id) != str(object_pose["frame_id"]):
                    raise ValueError("物体与末端位姿必须位于同一实时坐标系")
                values.append(
                    {
                        "tool_ref": ref,
                        "target_pose": {
                            "frame_id": current.frame_id,
                            "position_m": [
                                float(current.position[index]) + translation[index]
                                for index in range(3)
                            ],
                            "orientation_xyzw": list(current.quaternion_xyzw),
                            "observed_at": object_pose.get("observed_at"),
                            "revision": data["target_revision"],
                        },
                    }
                )
            return values

        clearance_poses = translated_tool_poses(clearance_delta)
        transport_poses = translated_tool_poses(delta)

        value = {
            "object_ref": data["object_ref"],
            "target_revision": data["target_revision"],
            "desired_object_pose": {
                **object_pose,
                "position_m": desired_center,
            },
            "clearance_poses": clearance_poses,
            "transport_poses": transport_poses,
        }
        obs = observation(
            "transport_posture",
            profile_name,
            str(data["object_ref"]),
            value,
            revision=str(data["target_revision"]),
            confidence=1.0,
        )
        return self.succeeded(
            "PlanTransportPosture",
            data,
            invocation_id,
            digest,
            value,
            (obs,),
        )


def _order_strategies_for_scene(
    ordered: list[str],
    data: Mapping[str, Any],
    settings: Mapping[str, Any],
    robot_base_pose: Pose,
    scene_objects: Sequence[Any],
) -> list[str]:
    """把没有邻箱阻挡的外侧候选排在前面。

    这里只使用 Runtime 实时对象 AABB 判断工具接近走廊是否被同层周转箱占用；
    它是 MuJoCo ground-truth 抓取 Provider 的环境碰撞输入，不进入 Robot Skill、
    SDK 或 Semantic Map。最终候选仍要通过 IK、自碰撞和真实接触验证。
    """

    pose = data["target_pose"]
    center = tuple(float(value) for value in pose["position_m"])
    extent = tuple(float(value) for value in data["object_extent_m"])
    groove_axis = _rotate_vector(
        pose["orientation_xyzw"],
        _normalized_axis(settings.get("groove_axis_local", [0.0, 1.0, 0.0])),
    )
    lane_axis = _normalized_axis((-groove_axis[1], groove_axis[0], 0.0))
    base_left = _rotate_vector(robot_base_pose.quaternion_xyzw, (0.0, 1.0, 0.0))
    left_sign = 1.0 if _dot(groove_axis, base_left) >= 0.0 else -1.0
    side_sign = {"left": left_sign, "right": -left_sign}
    blocked = {"left": False, "right": False}
    target_axis_span = sum(abs(groove_axis[i]) * extent[i] for i in range(3))
    target_lane_span = sum(abs(lane_axis[i]) * extent[i] for i in range(3))
    access_clearance = float(settings.get("approach_clearance_m", 0.08))

    for item in scene_objects:
        if str(getattr(item, "source_id", "")) == str(data["object_ref"]):
            continue
        if str(getattr(item, "category", "")) != "tote":
            continue
        other_extent_raw = getattr(item, "extent", None)
        other_pose = getattr(item, "pose", None)
        if other_extent_raw is None or other_pose is None:
            continue
        other_extent = tuple(float(value) for value in other_extent_raw)
        delta = tuple(float(other_pose.position[i]) - center[i] for i in range(3))
        # 上下层箱体会在Z方向相邻，但不会封堵同层的水平接近通道。
        if abs(delta[2]) >= (extent[2] + other_extent[2]) / 2.0 - 0.01:
            continue
        other_lane_span = sum(abs(lane_axis[i]) * other_extent[i] for i in range(3))
        if abs(_dot(lane_axis, delta)) >= (target_lane_span + other_lane_span) / 2.0:
            continue
        axis_projection = _dot(groove_axis, delta)
        if abs(axis_projection) < 1e-6:
            continue
        other_axis_span = sum(abs(groove_axis[i]) * other_extent[i] for i in range(3))
        surface_gap = abs(axis_projection) - (target_axis_span + other_axis_span) / 2.0
        if surface_gap >= access_clearance:
            continue
        for side, sign in side_sign.items():
            if axis_projection * sign > 0.0:
                blocked[side] = True

    if blocked["left"] == blocked["right"]:
        return ordered
    accessible = "right" if blocked["left"] else "left"
    inaccessible = "left" if accessible == "right" else "right"
    return [
        f"{accessible}_extract_first",
        "direct_bilateral",
        f"{inaccessible}_extract_first",
    ]



def _grasp_collision_environment(
    sdk: Any,
    candidate: Mapping[str, Any],
    *,
    exclude_source_ids: tuple[str, ...] = (),
    allowed_contact_end_effectors_by_source: Mapping[str, Sequence[str]] | None = None,
) -> EnvironmentCollisionSet | None:
    """为抓取预检保留中空箱壁和侧面凹槽，不把目标箱近似为实心块。"""

    kwargs: dict[str, Any] = {}
    if exclude_source_ids:
        kwargs["exclude_source_ids"] = exclude_source_ids
    if allowed_contact_end_effectors_by_source:
        kwargs["allowed_contact_end_effectors_by_source"] = (
            allowed_contact_end_effectors_by_source
        )
    environment = scene_collision_environment(sdk, **kwargs)
    if environment is None:
        return None

    object_ref = str(candidate.get("object_ref") or "")
    geometry = candidate.get("_target_collision_geometry")
    if not object_ref or not isinstance(geometry, Mapping):
        return environment
    target = next(
        (item for item in environment.objects if item.source_id == object_ref),
        None,
    )
    if target is None:
        return environment

    parts = _tote_collision_parts(target, geometry)
    if not parts:
        return environment
    return EnvironmentCollisionSet(
        frame_id=environment.frame_id,
        objects=[
            part
            for item in environment.objects
            for part in (parts if item.source_id == object_ref else [item])
        ],
    )


def _tote_collision_parts(
    target: EnvironmentCollisionObject,
    geometry: Mapping[str, Any],
) -> list[EnvironmentCollisionObject]:
    """用普通box表达开口箱壁和左右凹槽；尺寸不足时保留原始包络。"""

    extent = tuple(float(value) for value in geometry.get("extent_m", ()))
    axis = tuple(float(value) for value in geometry.get("groove_axis_local", ()))
    groove_width = float(geometry.get("groove_width_m", 0.0))
    groove_height = float(geometry.get("groove_height_m", 0.0))
    groove_depth = float(geometry.get("groove_depth_m", 0.0))
    top_offset = float(geometry.get("groove_top_offset_m", 0.0))
    wall = float(geometry.get("wall_thickness_m", 0.0))
    if len(extent) != 3 or len(axis) != 3:
        return []
    horizontal = [abs(axis[0]), abs(axis[1])]
    axis_index = 0 if horizontal[0] >= horizontal[1] else 1
    lane_index = 1 - axis_index
    if horizontal[axis_index] < 0.99:
        return []

    axis_span = extent[axis_index]
    lane_span = extent[lane_index]
    height = extent[2]
    groove_top = height / 2.0 - top_offset
    groove_bottom = groove_top - groove_height
    if (
        wall <= 0.0
        or groove_depth <= wall
        or groove_width <= wall
        or groove_width >= lane_span - 2.0 * wall
        or groove_bottom <= -height / 2.0 + wall
        or groove_top >= height / 2.0 - wall
    ):
        return []

    parts: list[EnvironmentCollisionObject] = []

    def append(
        name: str,
        local_position: Sequence[float],
        local_size: Sequence[float],
    ) -> None:
        world_offset = _rotate_vector(
            target.pose.quaternion_xyzw,
            tuple(float(value) for value in local_position),
        )
        parts.append(
            EnvironmentCollisionObject(
                source_id=f"{target.source_id}#collision/{name}",
                shape="box",
                pose=Pose(
                    frame_id=target.pose.frame_id,
                    position=tuple(
                        target.pose.position[index] + world_offset[index]
                        for index in range(3)
                    ),
                    quaternion_xyzw=target.pose.quaternion_xyzw,
                ),
                size_xyz=tuple(float(value) for value in local_size),
                allowed_contact_end_effectors=target.allowed_contact_end_effectors,
            )
        )

    def values(axis_value: float, lane_value: float, z_value: float) -> list[float]:
        result = [0.0, 0.0, z_value]
        result[axis_index] = axis_value
        result[lane_index] = lane_value
        return result

    def sizes(axis_value: float, lane_value: float, z_value: float) -> list[float]:
        result = [0.0, 0.0, z_value]
        result[axis_index] = axis_value
        result[lane_index] = lane_value
        return result

    append(
        "bottom",
        values(0.0, 0.0, -height / 2.0 + wall / 2.0),
        sizes(axis_span, lane_span, wall),
    )
    for lane_sign in (-1.0, 1.0):
        append(
            f"long-wall-{lane_sign:+.0f}",
            values(0.0, lane_sign * (lane_span - wall) / 2.0, 0.0),
            sizes(axis_span, wall, height),
        )

    side_width = (lane_span - groove_width) / 2.0
    below_height = groove_bottom + height / 2.0
    above_height = height / 2.0 - groove_top
    for axis_sign in (-1.0, 1.0):
        wall_axis = axis_sign * (axis_span - wall) / 2.0
        for lane_sign in (-1.0, 1.0):
            append(
                f"short-side-{axis_sign:+.0f}-{lane_sign:+.0f}",
                values(
                    wall_axis,
                    lane_sign * (groove_width / 2.0 + side_width / 2.0),
                    0.0,
                ),
                sizes(wall, side_width, height),
            )
        append(
            f"below-groove-{axis_sign:+.0f}",
            values(
                wall_axis,
                0.0,
                -height / 2.0 + below_height / 2.0,
            ),
            sizes(wall, groove_width, below_height),
        )
        append(
            f"above-groove-{axis_sign:+.0f}",
            values(
                wall_axis,
                0.0,
                groove_top + above_height / 2.0,
            ),
            sizes(wall, groove_width, above_height),
        )
        append(
            f"groove-backing-{axis_sign:+.0f}",
            values(
                axis_sign * (axis_span / 2.0 - groove_depth - wall / 2.0),
                0.0,
                (groove_bottom + groove_top) / 2.0,
            ),
            sizes(wall, groove_width, groove_height),
        )
        groove_axis_center = axis_sign * (axis_span / 2.0 - groove_depth / 2.0)
        append(
            f"groove-ceiling-{axis_sign:+.0f}",
            values(groove_axis_center, 0.0, groove_top + wall / 2.0),
            sizes(groove_depth, groove_width, wall),
        )
        append(
            f"groove-floor-{axis_sign:+.0f}",
            values(groove_axis_center, 0.0, groove_bottom - wall / 2.0),
            sizes(groove_depth, groove_width, wall),
        )
        for lane_sign in (-1.0, 1.0):
            append(
                f"groove-end-{axis_sign:+.0f}-{lane_sign:+.0f}",
                values(
                    groove_axis_center,
                    lane_sign * (groove_width + wall) / 2.0,
                    (groove_bottom + groove_top) / 2.0,
                ),
                sizes(groove_depth, wall, groove_height),
            )
    return parts

def _candidate_reachable(
    sdk: Any,
    candidate: Mapping[str, Any],
    robot_state: Any,
    tools: Mapping[str, Any],
) -> bool:
    reachable, _ = _candidate_reachability(
        sdk, candidate, robot_state, tools, object_ref=str(candidate["object_ref"])
    )
    return reachable


def _candidate_reachability(
    sdk: Any,
    candidate: dict[str, Any],
    robot_state: Any,
    tools: Mapping[str, Any],
    *,
    engaged_tool_ref: str | None = None,
    object_ref: str,
    secondary_resume_phase: str | None = None,
) -> tuple[bool, str | None]:
    """按实际Robot状态淘汰无法执行的几何候选。

    direct_bilateral会连续验证接近、插入和抬升。单侧外拉改变目标物位姿，
    外拉后Skill必须重新观测并重新规划，因此这里只验证重观测前的首侧动作，
    不能拿旧Pose提前验证第二侧或抬升。
    """

    side_by_ref = {tool.tool_ref: side for side, tool in tools.items()}
    strategy = str(candidate["strategy"])
    fixed_targets: dict[str, Pose] = {}
    validate_motion_paths = False
    # 目标箱在clearance/transfer/pregrasp期间仍是运动侧的真实障碍。首侧
    # 已经钩入凹槽时，只允许该固定末端分支保持既有接触；不能像旧实现那样
    # 排除整个箱体，否则第二只手的轨迹会穿过箱体并把它从首侧钩脚上推走。
    environment = _grasp_collision_environment(sdk, candidate)
    if engaged_tool_ref is not None:
        if engaged_tool_ref not in side_by_ref:
            return False, f"已接合工具不属于当前Robot: {engaged_tool_ref}"
        engaged_side = side_by_ref[engaged_tool_ref]
        environment = _grasp_collision_environment(
            sdk,
            candidate,
            allowed_contact_end_effectors_by_source={
                object_ref: (engaged_side,)
            },
        )
        secondary_side = "right" if engaged_side == "left" else "left"
        engaged_target = robot_state.end_effectors.get(engaged_side)
        if engaged_target is None:
            return False, f"Robot状态缺少已接合的{engaged_side}工具位姿"
        # 外拉后只预检第二只手下一段clearance。它位于箱体上方自由空间，
        # 应与真实Action一样先走单臂局部解；把首侧末端钉死为第二个IK目标会
        # 在动作尚未开始前制造多末端不收敛。首侧既有接触仍进入环境白名单，
        # 真正执行时由ManipulatorMotion连续监测，丢失即stop+hold。
        secondary_groups = [
            (
                f"secondary_{phase}",
                [
                    item
                    for item in candidate[f"{phase}_poses"]
                    if side_by_ref[item["tool_ref"]] == secondary_side
                ],
            )
            # 首侧接合后，第二侧依次进入身体前方走廊、箱体前缘高位、
            # 凹槽中心上方和槽口外侧。每段从上一段真实终点继续预检，
            # 不能用一个大中点逼迫IK选择绕背分支。
            for phase in (
                "clearance",
                "transfer",
                "alignment",
                "pregrasp",
                "approach",
            )
        ]
        # 这个阶段来自同一Skill Execution已经完成的真实Action。第二手已到
        # 槽口外并重新观测时，只需要更新后续insert/seat目标；不能把已经
        # 执行成功的clearance/transfer/pregrasp再当作新路径预检一次。
        if secondary_resume_phase == "insert":
            secondary_groups = []

        # 未显式指定恢复阶段时，才根据实时末端位置跳过已经到达的候选节点；
        # 显式insert来自同一Execution的完成记录，优先级高于位置容差推断。
        # 不能让已经在槽口外的手臂返回clearance再走一遍。
        # 容差由夹具插入行程和固定端跟踪容差共同推导，不是场景帧数。
        resume_tolerance = float(
            candidate.get("_secondary_resume_tolerance_m", 0.0)
        )
        if (
            secondary_resume_phase is None
            and secondary_groups
            and resume_tolerance > 0.0
        ):
            secondary_current = robot_state.end_effectors.get(secondary_side)
            if secondary_current is not None:
                completed_group_index: int | None = None
                for group_index in range(len(secondary_groups) - 1, -1, -1):
                    group = secondary_groups[group_index][1]
                    if len(group) != 1:
                        continue
                    position = group[0]["target_pose"]["position_m"]
                    if math.dist(
                        tuple(float(value) for value in secondary_current.position),
                        tuple(float(value) for value in position),
                    ) <= resume_tolerance:
                        completed_group_index = group_index
                        break
                if completed_group_index is not None:
                    secondary_groups = secondary_groups[completed_group_index + 1 :]

        matching_extract_strategy = f"{engaged_side}_extract_first"
        if strategy == matching_extract_strategy:
            needs_more_clearance = (
                candidate.get("_secondary_preflight_mode")
                == "additional_extraction"
            )
            secondary_reason = "第二侧入口净空仍不足"
            if not needs_more_clearance:
                # 候选生成只确认下一段动作能够启动，不能在第一侧已经承载时
                # 一次性模拟 clearance 到 approach 的全部未来路径。后续各段
                # 会从真实执行终点重新规划，并由 ManipulatorMotion 对实际轨迹
                # 做碰撞、固定承载侧和接触检查。把尚未发生的 alignment IK 或
                # 假设关节插值碰撞提前当成整个抓取不可达，会让已经成功外拉的
                # 现场无故进入 waiting_agent，也会诱导继续补拉或叠加回退路径。
                next_groups = secondary_groups[:1]
                secondary_reachable, secondary_reason = _preflight_candidate_groups(
                    sdk,
                    candidate,
                    robot_state,
                    side_by_ref,
                    next_groups,
                    fixed_targets,
                    environment,
                    validate_motion_paths=True,
                )
                if secondary_reachable:
                    candidate["pull_path"] = []
                    candidate["strategy"] = "direct_bilateral"
                    candidate["candidate_id"] = (
                        "direct_bilateral-"
                        + candidate["candidate_id"].rsplit("-", 1)[-1]
                    )
                    return True, None
                # 下一段实际动作本身不可规划时才拒绝当前候选。几何净空已经
                # 满足后不继续用外拉掩盖路径问题，也不放宽真实碰撞边界。
                return False, secondary_reason
            if not candidate.get("pull_path"):
                return False, secondary_reason
            candidate["_deferred_secondary_reason"] = secondary_reason
            groups = [("additional_extract", candidate["pull_path"])]
            # 补拉动作本身要移动已经接合的首侧，不能同时把该末端标为固定。
            # 该动作与目标箱已有预期接触，因此只在这一条路径中排除目标箱。
            fixed_targets = {}
            environment = _grasp_collision_environment(
                sdk, candidate, exclude_source_ids=(object_ref,)
            )
            validate_motion_paths = False
        else:
            groups = secondary_groups
            validate_motion_paths = True
    elif strategy == "direct_bilateral":
        # 候选阶段只确认下一段 clearance 能够启动。transfer/approach 会从
        # clearance 的真实终点重新规划；提前把三段未来路径全部求解一遍，
        # 不仅重复实际 Action 的 IK/碰撞工作，还会用尚未发生的假设姿态把
        # 可执行候选过早淘汰。真实下发轨迹仍逐点检查碰撞。
        groups = [
            ("bilateral_clearance", candidate["clearance_poses"]),
        ]
    else:
        first_side = strategy.removesuffix("_extract_first")
        def first_side_group(phase: str) -> list[Mapping[str, Any]]:
            primary = next(
                item
                for item in candidate[f"{phase}_poses"]
                if side_by_ref[item["tool_ref"]] == first_side
            )
            # 首侧尚未形成物理承载时，只规划工作臂。非工作臂保持当前自然
            # travel/安全姿态；SDK仍使用它的当前关节参与整机碰撞检查。
            return [primary]

        groups = [
            ("first_primary_clearance", first_side_group("clearance")),
        ]
        # 首侧尚未接触箱体时不需要在候选生成阶段重复规划完整轨迹；这里只
        # 做 clearance 终点 IK，实际 Action 再验证连续路径。外拉后的第二侧
        # 是否已经获得完整入口仍由上面的 engaged_tool_ref 分支严格判断。
        validate_motion_paths = False

    # 候选预检只判断进入当前策略前的无接触路径。insert/seat 必须以实时
    # 接触终止；外拉会改变物体 Pose，transport 又依赖真正形成的抓取关系。
    # 用当前理想状态提前验证这些未来阶段会把可执行候选错误过滤掉。每个后续
    # Action 仍会依据最新 Robot 状态独立做 IK、碰撞、接触力和停止检查。

    return _preflight_candidate_groups(
        sdk,
        candidate,
        robot_state,
        side_by_ref,
        groups,
        fixed_targets,
        environment,
        validate_motion_paths=validate_motion_paths,
    )


def _preflight_candidate_groups(
    sdk: Any,
    candidate: Mapping[str, Any],
    robot_state: Any,
    side_by_ref: Mapping[str, str],
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    fixed_targets: Mapping[str, Pose],
    environment: Any,
    *,
    validate_motion_paths: bool = False,
) -> tuple[bool, str | None]:
    """从同一实时状态连续预检一组无接触运动。"""

    trial_state = robot_state
    for group_index, (phase, group) in enumerate(groups):
        targets = dict(fixed_targets)
        for item in group:
            raw = item["target_pose"]
            targets[side_by_ref[item["tool_ref"]]] = Pose(
                position=tuple(float(value) for value in raw["position_m"]),
                quaternion_xyzw=tuple(
                    float(value) for value in raw["orientation_xyzw"]
                ),
                frame_id=str(raw["frame_id"]),
            )
        # 候选预检必须与真实Action使用相同的固定端约束。首侧下钩已经
        # 进入凹槽后，第二侧的clearance、transfer、alignment和pregrasp都
        # 不能借共享躯干拖动首侧；否则预检会报告可达，执行时却只能在
        # “保持钩入”和“命中第二侧目标”之间二选一。
        path_fixed_targets = frozenset(fixed_targets)
        try:
            solution: Mapping[str, float] | None = None
            if validate_motion_paths:
                # 外拉何时结束必须由第二侧整条进入路径决定，不能只验证第一个
                # clearance终点。箱体前上方到侧槽上方仍属于无接触自由空间，
                # 采用连续关节路径并逐点检查碰撞；只有侧槽上方到槽口外侧的
                # 垂直下降保持短笛卡尔路径。强迫前一段做直线笛卡尔插值会在
                # 固定首侧时制造并不存在的中间IK不可达。
                purpose = phase.rsplit("_", 1)[-1]
                # clearance/transfer属于无遮挡自由空间：使用连续关节时间
                # 轨迹，并由SDK逐点投影固定承载侧、检查Robot和环境碰撞。
                # 若强制长距离笛卡尔直线，每个中间点都只能沿局部IK分支，
                # 会把终点可达的自然路径误报为不收敛。只有槽口上方下降和
                # 进入凹槽前的短路径保持笛卡尔几何。
                preserve_cartesian_path = purpose in {"pregrasp", "approach"}
                maximum_speed_mps = {
                    "clearance": 0.08,
                    "transfer": 0.10,
                    "alignment": 0.10,
                    "pregrasp": 0.10,
                    "approach": 0.08,
                }.get(purpose, 0.08)
                if len(targets) == 1:
                    end_effector, target = next(iter(targets.items()))
                    planned = sdk.upper_body.plan_end_effector(
                        end_effector,
                        target,
                        state=trial_state,
                        environment=environment,
                        speed_scale=1.0,
                        maximum_speed_mps=maximum_speed_mps,
                        preserve_cartesian_path=preserve_cartesian_path,
                    )
                else:
                    planned = sdk.upper_body.plan_end_effectors(
                        targets,
                        state=trial_state,
                        environment=environment,
                        speed_scale=1.0,
                        maximum_speed_mps=maximum_speed_mps,
                        preserve_cartesian_path=preserve_cartesian_path,
                        fixed_end_effectors=path_fixed_targets,
                    )
                if not planned.joint_trajectory:
                    raise PlanningError(f"{phase} 没有生成关节轨迹")
                # 下一段必须从本段完整路径真正选择的终点继续预检。若这里再次
                # 单独调用IK，求解器可能切到另一个合法分支，后续碰撞诊断就不再
                # 对应实际clearance→transfer→pregrasp动作，还会重复最昂贵的计算。
                solution = planned.joint_trajectory[-1].positions
            if solution is None:
                solution = sdk.kinematics_provider.solve_many(
                    robot_id=sdk.robot_id,
                    targets=targets,
                    state=trial_state,
                    environment=environment,
                    fixed_end_effectors=path_fixed_targets,
                )
        except PlanningError as error:
            # 精确阶段和底层原因用于调整工位或策略，不增加新的执行协议。
            # 只在规划尚未产生任何物理动作时附带当前关节和末端目标，便于
            # 区分目标工作区错误与IK分支错误；这些诊断不是Skill输入或状态。
            strategy = str(candidate["strategy"])
            target_positions = {
                name: [round(float(value), 4) for value in target.position]
                for name, target in targets.items()
            }
            current_positions = {
                name: [round(float(value), 4) for value in pose.position]
                for name, pose in trial_state.end_effectors.items()
                if name in targets
            }
            joint_positions = {
                name: round(float(value), 4)
                for name, value in trial_state.joint_positions.items()
                if name.startswith(("torso_joint", "left_arm_joint", "right_arm_joint"))
            }
            return False, (
                f"{strategy}/{phase}: {error}; "
                f"targets={target_positions}; current={current_positions}; "
                f"joints={joint_positions}"
            )
        joints = dict(trial_state.joint_positions)
        joints.update(solution)
        end_effectors = dict(trial_state.end_effectors)
        end_effectors.update(targets)
        trial_state = trial_state.model_copy(
            update={"joint_positions": joints, "end_effectors": end_effectors}
        )
    return True, None


def _candidate(
    strategy: str,
    index: int,
    data: Mapping[str, Any],
    tools: Mapping[str, Any],
    settings: Mapping[str, Any],
    robot_state: Any,
    scene_objects: Sequence[Any] = (),
    *,
    motion_position_tolerance_m: float = 1e-4,
) -> dict[str, Any]:
    robot_base_pose = robot_state.base_pose
    if robot_base_pose is None:
        raise ValueError("抓取候选需要Robot实时底盘位姿")
    pose = data["target_pose"]
    center = [float(value) for value in pose["position_m"]]
    extent = [float(value) for value in data["object_extent_m"]]
    groove_axis_local = _normalized_axis(
        settings.get("groove_axis_local", [0.0, 1.0, 0.0])
    )
    groove_axis_world = _rotate_vector(pose["orientation_xyzw"], groove_axis_local)
    lane_axis_world = _normalized_axis(
        (-groove_axis_world[1], groove_axis_world[0], 0.0)
    )
    # 箱体左右短边各有一条连续凹槽，不存在承力条或中央安装块。抓取点无需
    # 固定在凹槽正中心：在完整钩脚仍位于槽内的前提下，选择靠Robot的一段
    # 可以显著缩短前伸距离，避免高层箱体迫使躯干大幅偏航。横向实体尺寸
    # 来自夹具Profile；未声明时保守退回凹槽中心。
    groove_width = float(settings.get("groove_width_m", 0.0))
    tool_lateral_envelope = float(settings.get("tool_lateral_envelope_m", 0.0))
    wall_thickness = float(settings.get("wall_thickness_m", 0.0))
    if groove_width < 0.0 or tool_lateral_envelope < 0.0 or wall_thickness < 0.0:
        raise ValueError("凹槽与夹具横向尺寸不能小于零")
    if tool_lateral_envelope > groove_width > 0.0:
        raise ValueError("夹具横向实体宽度不能大于抓取凹槽宽度")
    base_lane_delta = _dot(
        lane_axis_world,
        tuple(robot_base_pose.position[index] - center[index] for index in range(3)),
    )
    front_sign = 1.0 if base_lane_delta >= 0.0 else -1.0
    lane_offset = 0.0
    if groove_width > 0.0 and tool_lateral_envelope > 0.0:
        lane_offset = front_sign * max(
            0.0,
            (groove_width - tool_lateral_envelope) / 2.0 - wall_thickness,
        )

    base_left_world = _rotate_vector(robot_base_pose.quaternion_xyzw, (0.0, 1.0, 0.0))
    left_sign = 1.0 if _dot(groove_axis_world, base_left_world) >= 0.0 else -1.0
    side_sign = {"left": left_sign, "right": -left_sign}
    groove_axis_span = sum(abs(groove_axis_local[i]) * extent[i] for i in range(3))
    # Runtime返回的是物体局部几何尺寸；物体旋转后，必须先把待投影的
    # 世界方向变换回物体局部系。直接用世界轴乘局部长宽会在15度偏转时
    # 把400mm箱深错误放大到约542mm，进而制造异常外拉和绕行路径。
    target_axis_span = _extent_span_along_world_axis(
        extent, pose["orientation_xyzw"], groove_axis_world
    )
    target_lane_span = _extent_span_along_world_axis(
        extent, pose["orientation_xyzw"], lane_axis_world
    )
    clearance = float(settings.get("approach_clearance_m", 0.08))
    groove_depth = float(settings["groove_depth_m"])
    groove_backing_clearance = float(settings["groove_backing_clearance_m"])
    hook_foot_length = float(settings["hook_foot_length_m"])
    hook_forward_reach = float(settings["hook_forward_reach_from_load_frame_m"])
    hook_upper_reach = float(settings["hook_upper_reach_from_load_frame_m"])
    hook_foot_thickness = float(settings["hook_foot_thickness_m"])
    engagement_travel = float(settings["hook_engagement_travel_m"])
    if min(
        groove_depth,
        groove_backing_clearance,
        hook_foot_length,
        hook_forward_reach,
        hook_upper_reach,
        hook_foot_thickness,
        engagement_travel,
    ) <= 0.0:
        raise ValueError("凹槽和钩脚几何尺寸必须为正数")
    if hook_forward_reach > hook_foot_length:
        raise ValueError("钩脚前伸量不能大于钩脚总长度")
    # 抓取目标控制的是 load frame，不是钩脚几何中心。插入深度必须扣除
    # load frame 到钩脚最前端的真实距离；否则钩尖会撞到凹槽后壁并把箱体
    # 向内推开，看起来像“已经接触”，实际却没有进入两侧凹槽。
    insertion = groove_depth - hook_forward_reach - groove_backing_clearance
    if insertion <= 0.0:
        raise ValueError("抓取凹槽深度不足以让钩脚进入并避开背板")
    # C钩的竖直钩脊必须留在箱体外侧，水平钩脚从槽口跨入凹槽；不能要求
    # “整根钩脚越过槽口”。后者会把钩脊也推入箱壁，表现为双臂在箱体
    # 外表面对称受阻。实际插入路径仍由SDK碰撞预检，最终接合由实时接触、
    # 支撑方向和箱体跟随运动确认。
    # 夹具开合行程沿工具Z轴，不能叠加到箱体侧面的水平进入宽度。旧算法把
    # 39mm开合行程当成径向包络，密集堆叠时会凭空要求十几厘米外拉。这里的
    # access envelope只描述工具通过侧向通道所需的真实宽度，由夹具Profile
    # 给出；未配置时以实际插钩深度作为保守近似，最终仍由SDK实体碰撞预检。
    tool_access_envelope = float(settings.get("tool_access_envelope_m", insertion))
    # 工具参考系通常不位于径向包络的几何中心。目标侧尺寸来自同一夹具
    # Provider的物理接口配置；邻箱侧尺寸由完整包络扣除得到，避免复制URDF
    # 碰撞体。未配置的其他工具仍退化为对称包络。
    tool_target_side_envelope = float(
        settings.get("tool_target_side_envelope_m", tool_access_envelope / 2.0)
    )
    if not 0.0 <= tool_target_side_envelope <= tool_access_envelope:
        raise ValueError("tool_target_side_envelope_m 必须位于完整工具包络内")
    # 实际入口除了工具实体宽度，还要容纳RobotDeployment允许的固定端
    # 跟踪误差和感知到的箱体微小偏转。这个余量属于工具路径Profile，不是
    # 箱型经验或调用方参数；达到该几何门槛后仍必须通过SDK整条路径碰撞预检。
    tool_path_clearance_margin = float(
        settings.get("tool_path_clearance_margin_m", 0.0)
    )
    if tool_path_clearance_margin < 0.0:
        raise ValueError("tool_path_clearance_margin_m 不能小于零")
    secondary_entry_gap = tool_access_envelope + tool_path_clearance_margin
    # 首次外拉要一次形成可供第二侧规划的工作通道，避免刚夹紧后重复克服
    # 静摩擦；它使用Provider已有的接近净空。外拉后的重观测只检查夹具
    # 实体入口，腕部、相机和前臂则交给SDK完整路径碰撞预检，不能继续用
    # 12cm通道追逐毫米尾差。
    initial_extraction_gap = max(secondary_entry_gap, clearance)
    # 第二侧现在从箱体前方高位走廊进入，再横向对准侧壁凹槽。侧向箱间隙
    # 只需先容纳真实工具包络，腕部和前臂是否可通过必须交给SDK整条路径的
    # 实体碰撞预检。若仍明确撞到同侧邻箱，再按本次碰撞证据继续外拉；不能
    # 在预检前强迫箱体先腾出完整approach clearance，否则机械臂到达外拉
    # 边界时会为了几毫米的几何尾差丢钩。
    secondary_path_gap = secondary_entry_gap
    pregrasp_opening = float(settings.get("pregrasp_opening_m", 0.035))
    clamp_target_position = float(settings.get("clamp_target_position_m", 0.015))
    groove_top_offset = float(settings["groove_top_offset_m"])
    if groove_top_offset <= 0.0:
        raise ValueError("groove_top_offset_m 必须为正数")
    # SceneSnapshot给出箱体中心。工具参考系位于夹具Profile定义的load frame，
    # 水平插入时必须让钩脚最上方也完整进入凹槽；若只按钩脚厚度的一半
    # 计算，向上的钩角会撞在槽口上方，Robot只能把箱体推开。seat再从这个
    # 低位短上移，使钩角贴住凹槽内上表面。
    object_top = center[2] + extent[2] / 2.0
    load_height = object_top - groove_top_offset - hook_upper_reach

    # transfer只需让夹具钩脚最低点越过当前箱体顶部，而不是无条件在钩脚上方
    # 再抬20cm。后者会使高层箱体的准备位姿超出Robot工作空间。这里由实时箱体
    # 几何和夹具Provider的越障余量推导高度；Runtime、SDK和Skill都不感知这条
    # 周转箱抓取策略。该余量描述夹具实体尺寸，不是调用方可调的业务参数。
    transfer_clearance = float(settings.get("transfer_clearance_above_top_m", 0.02))
    safe_transfer_offset = max(0.0, object_top + transfer_clearance - load_height)

    def at(
        side: str,
        *,
        radial_offset: float = 0.0,
        extra_z: float = 0.0,
        lane_offset_override: float | None = None,
    ) -> dict[str, Any]:
        distance = groove_axis_span / 2.0 + radial_offset
        sign = side_sign[side]
        # 候选直接对准Scene Package公开的侧面凹槽中心。Runtime只采集
        # 当前接触、力和相对速度，Ability随后按动作语义判断是否完成就位；
        # Provider不再为不存在的内部零件维护横向绕行偏移。
        effective_lane_offset = (
            lane_offset if lane_offset_override is None else lane_offset_override
        )
        position = [
            center[i]
            + groove_axis_world[i] * sign * distance
            + lane_axis_world[i] * effective_lane_offset
            for i in range(3)
        ]
        position[2] = load_height + extra_z
        return {
            "frame_id": pose["frame_id"],
            "position_m": position,
            # tote_clamp 的局部 +X 是钩爪插入方向。左右工具位于箱体相反两侧，
            # 因此必须分别朝向箱体中心；若两侧都复制底盘朝向，工具参考系的
            # 位置虽然可达，实体钩爪却会与抓取凹槽正交，最终永远得不到接触证据。
            # 该朝向是夹具 Provider 的实时几何结果，不进入通用 SDK 或 Skill。
            "orientation_xyzw": _inward_tool_orientation(groove_axis_world, sign),
            "observed_at": pose.get("observed_at"),
            "revision": data["target_revision"],
        }

    tool_targets = [
        {"tool_ref": tools[side].tool_ref, "side": side} for side in ("left", "right")
    ]
    # 首侧从当前自然待机位直接向箱体外侧高位移动。旧路径先在当前XY垂直
    # 抬到箱顶，再横移近一个工作空间：高层箱会迫使躯干大幅后仰、扭腰，
    # 非工作臂也会随共享躯干扫向箱体。凹槽位于箱体侧面，工具在箱外运动
    # 时无需先完成这次垂直越障。这里取当前末端与外侧transfer目标的空间
    # 中点，只改变位置并保持当前腕姿；随后transfer才切换为钩入朝向。
    clearance_poses = []
    for side in ("left", "right"):
        current = robot_state.end_effectors.get(side)
        if current is None:
            raise ValueError(f"Robot状态缺少{side}末端位姿")
        transfer_target = at(
            side,
            radial_offset=clearance,
            extra_z=safe_transfer_offset,
        )
        clearance_poses.append(
            {
                "tool_ref": tools[side].tool_ref,
                "target_pose": {
                    "frame_id": current.frame_id,
                    "position_m": [
                        (
                            float(current.position[index])
                            + float(transfer_target["position_m"][index])
                        )
                        / 2.0
                        for index in range(3)
                    ],
                    "orientation_xyzw": list(current.quaternion_xyzw),
                    "observed_at": pose.get("observed_at"),
                    "revision": data["target_revision"],
                },
            }
        )
    if not data.get("engaged_tool_ref") and strategy.endswith("_extract_first"):
        first_side = strategy.removesuffix("_extract_first")
        ready_side = "right" if first_side == "left" else "left"
        first_clearance = next(
            item
            for item in clearance_poses
            if item["tool_ref"] == tools[first_side].tool_ref
        )
        first_current = robot_state.end_effectors[first_side]
        first_clearance_target = at(
            first_side,
            radial_offset=clearance,
            extra_z=safe_transfer_offset,
            lane_offset_override=front_sign * (target_lane_span / 2.0 + clearance),
        )
        first_clearance["target_pose"] = {
            **first_clearance_target,
            "orientation_xyzw": list(first_current.quaternion_xyzw),
        }

        # 准备手保留上面由“当前自然末端 → 未来箱外高位”推导出的
        # 几何中间位，不覆盖成尚未打开的对侧凹槽上方。该位置靠近身体、
        # 保持当前腕姿且不进入左右箱间通道；外拉并重新观测后，Provider
        # 才会从这个中间位生成到新露出凹槽的短路径。
    engaged_side = next(
        (
            side
            for side, tool in tools.items()
            if tool.tool_ref == data.get("engaged_tool_ref")
        ),
        None,
    )
    secondary_side: str | None = None
    secondary_compact_transfer: dict[str, Any] | None = None
    secondary_front_transfer: dict[str, Any] | None = None
    secondary_lane_offset_override: float | None = None
    approach_clearance_by_side = {"left": clearance, "right": clearance}
    available_gap = math.inf
    secondary_preflight_mode = "path"
    secondary_obstacle_ids: list[str] = []
    if engaged_side is not None:
        secondary_side = "right" if engaged_side == "left" else "left"
        for item in scene_objects:
            if str(getattr(item, "source_id", "")) == str(data["object_ref"]):
                continue
            if str(getattr(item, "category", "")) != "tote":
                continue
            other_pose = getattr(item, "pose", None)
            other_extent_raw = getattr(item, "extent", None)
            if other_pose is None or other_extent_raw is None:
                continue
            other_extent = tuple(float(value) for value in other_extent_raw)
            delta = tuple(float(other_pose.position[i]) - center[i] for i in range(3))
            if abs(delta[2]) >= (extent[2] + other_extent[2]) / 2.0 - 0.01:
                continue
            other_lane_span = _extent_span_along_world_axis(
                other_extent, other_pose.quaternion_xyzw, lane_axis_world
            )
            if (
                abs(_dot(lane_axis_world, delta))
                >= (target_lane_span + other_lane_span) / 2.0
            ):
                continue
            axis_projection = _dot(groove_axis_world, delta)
            if axis_projection * side_sign[secondary_side] <= 0.0:
                continue
            secondary_obstacle_ids.append(str(getattr(item, "source_id", "")))
            other_axis_span = _extent_span_along_world_axis(
                other_extent, other_pose.quaternion_xyzw, groove_axis_world
            )
            surface_gap = (
                abs(axis_projection) - (target_axis_span + other_axis_span) / 2.0
            )
            available_gap = min(available_gap, max(0.0, surface_gap))
        if math.isfinite(available_gap):
            # 第二侧工具的目标是目标箱凹槽，不是两箱间隙的中心。工具参考系
            # 只保留夹具Profile声明的目标侧实体包络；外拉新增的其余净空全部
            # 留给腕部和前臂。把load frame居中会无意义地把link7推向邻箱，
            # 即使18cm通道也可能碰撞。精确工具、Robot和邻箱碰撞仍由SDK检查。
            available_approach_offset = tool_access_envelope
            approach_clearance_by_side[secondary_side] = min(
                clearance, available_approach_offset
            )
            # 完整工具包络表示实体宽度，目标侧包络为两侧共享的最小进入余量。
            # 达到由真实夹具尺寸推导的入口宽度后再做完整Robot路径预检。
            # 几何缺口小于当前SDK允许的末端位置误差时，继续生成亚毫米
            # 外拉只会让受限IK追逐数值尾差。此处仅切换为“先验证完整路径”；
            # SDK仍会用真实碰撞体检查整条右臂路径，绝不把这个容差当作免碰撞
            # 或抓取成功条件。路径不可达时返回真实IK/碰撞原因，而不是盲拉。
            if available_gap + motion_position_tolerance_m < secondary_path_gap:
                secondary_preflight_mode = "additional_extraction"
        secondary_current = robot_state.end_effectors[secondary_side]
        secondary_clearance = next(
            item
            for item in clearance_poses
            if item["tool_ref"] == tools[secondary_side].tool_ref
        )
        # 箱体左右短边的凹槽在lane方向连续贯通。第二侧已经由外拉打开
        # 径向通道后，最短且最自然的路线是在槽中心外侧升到箱顶以上，随后
        # 原地转腕并沿侧壁竖直下降；不需要先绕到箱体前角。旧的“前方
        # clearance”把高位目标推到肩部近端盲区，迫使IK扭腰、绕背，并且
        # 错误暗示夹具还要绕过某个不存在的承力构件。这里直接使用实时凹槽
        # 中心线；真实箱沿、邻箱和Robot碰撞仍由SDK对整条路径逐点检查。
        front_lane_offset = lane_offset
        if data.get("secondary_resume_phase") == "insert":
            # 第二只手已经沿真实轨迹到达槽口外，重观测只应修正箱体的径向
            # 位移。两侧凹槽在lane方向连续，若再次按底盘位置选择一个新的
            # 横向抓取点，会让末端先横移约一厘米再插入，并可能把一个已经
            # 可达的现场变成多末端IK无解。这里保留当前末端在凹槽方向上的
            # 位置，并把它裁剪到“完整夹具仍位于槽内”的几何范围；实际短
            # 插入仍由SDK做碰撞检查，接合仍由实时接触和受力证明。
            maximum_lane_offset = max(
                0.0,
                (groove_width - tool_lateral_envelope) / 2.0 - wall_thickness,
            )
            current_lane_offset = _dot(
                lane_axis_world,
                tuple(
                    float(secondary_current.position[index]) - center[index]
                    for index in range(3)
                ),
            )
            current_lane_offset = max(
                -maximum_lane_offset,
                min(maximum_lane_offset, current_lane_offset),
            )
            lane_step = maximum_lane_offset / 8.0
            lane_options = [
                current_lane_offset,
                max(-maximum_lane_offset, current_lane_offset - lane_step),
                min(maximum_lane_offset, current_lane_offset + lane_step),
            ]
            lane_index = int(data.get("_secondary_lane_candidate_index", 0))
            front_lane_offset = lane_options[min(lane_index, len(lane_options) - 1)]
            secondary_lane_offset_override = front_lane_offset
        secondary_front_transfer = at(
            secondary_side,
            radial_offset=approach_clearance_by_side[secondary_side],
            extra_z=safe_transfer_offset,
            lane_offset_override=front_lane_offset,
        )
        # layout001历史成功Execution采用三段连续走廊：先向前并抬高一个
        # 工具净空，再进入箱前安全高位，最后只保留一个净空的短alignment。
        # 这避免从身体附近一次斜穿到箱前而让腕部扫入相邻箱，也不增加固定
        # 关节模板；每段真实轨迹仍由SDK逐点检查碰撞和首侧接触。
        clearance_delta = [
            float(secondary_front_transfer["position_m"][index])
            - float(secondary_current.position[index])
            for index in range(3)
        ]
        clearance_horizontal_distance = math.hypot(
            clearance_delta[0], clearance_delta[1]
        )
        clearance_horizontal_progress = min(clearance, clearance_horizontal_distance)
        clearance_ratio = (
            clearance_horizontal_progress / clearance_horizontal_distance
            if clearance_horizontal_distance > 1e-9
            else 1.0
        )
        clearance_position = [
            float(secondary_current.position[index])
            + clearance_delta[index] * clearance_ratio
            for index in range(3)
        ]
        clearance_position[2] = float(secondary_current.position[2]) + clearance

        transfer_horizontal_progress = max(
            clearance_horizontal_progress,
            clearance_horizontal_distance - clearance - tool_access_envelope,
        )
        # 最后的高位alignment同时保留接近净空和夹具包络，避免把第二臂
        # 先推到工作空间边缘再要求一个精确短步。该余量来自工具Profile，
        # 会随夹具尺寸变化；不是为某个箱型写死的关节或位姿参数。
        transfer_ratio = (
            min(1.0, transfer_horizontal_progress / clearance_horizontal_distance)
            if clearance_horizontal_distance > 1e-9
            else 1.0
        )
        compact_transfer_position = [
            float(secondary_current.position[index])
            + clearance_delta[index] * transfer_ratio
            for index in range(3)
        ]
        transfer_height = float(secondary_front_transfer["position_m"][2])
        if secondary_side == "left":
            # R1 Pro左臂在固定右侧承载时，先前伸到低一个接近净空的位置，
            # 再由alignment同步抬高；提前高举会把肩肘推入不可解分支。
            # 右臂接入则直接走高位，避免把后续双侧抬升留在坏分支。两条
            # 中间路径都来自layout001真实成功Execution，终点和安全校验一致。
            transfer_height = max(
                float(secondary_current.position[2]), transfer_height - clearance
            )
        compact_transfer_position[2] = transfer_height
        secondary_clearance["target_pose"] = {
            "frame_id": secondary_current.frame_id,
            "position_m": clearance_position,
            "orientation_xyzw": list(secondary_front_transfer["orientation_xyzw"]),
            "observed_at": pose.get("observed_at"),
            "revision": data["target_revision"],
        }
        secondary_compact_transfer = {
            **secondary_clearance["target_pose"],
            "position_m": compact_transfer_position,
            "orientation_xyzw": list(
                secondary_front_transfer["orientation_xyzw"]
            ),
        }
    # 真实夹具从收拢姿态直接下探到箱沿高度时，钩体会在末端到位前扫过箱沿。
    # 候选因此显式给出“箱体外侧高位转移 → 外侧垂直下降 → 水平插入”三段短动作；
    # 这属于周转箱抓取 Provider 的几何策略，不下沉到通用 Robot SDK。
    transfer = [
        {
            "tool_ref": tools[side].tool_ref,
            "target_pose": at(
                side,
                radial_offset=approach_clearance_by_side[side],
                extra_z=safe_transfer_offset,
            ),
        }
        for side in ("left", "right")
    ]
    if secondary_side is not None and secondary_compact_transfer is not None:
        next(
            item
            for item in transfer
            if item["tool_ref"] == tools[secondary_side].tool_ref
        )["target_pose"] = secondary_compact_transfer
    # 第二侧先在箱体前上方完成前伸，再平移到侧壁凹槽正上方。前一个点
    # 避开相邻箱体，后一个点保证随后的下降是垂直短路径；二者不能合并为
    # 一条斜线，否则下钩会扫过目标箱长壁。它们都是同一抓取Stage的内部
    # 动作，不形成新的业务步骤。
    pregrasp = [
        {
            "tool_ref": tools[side].tool_ref,
            "target_pose": at(
                side,
                radial_offset=approach_clearance_by_side[side],
                extra_z=safe_transfer_offset,
            ),
        }
        for side in ("left", "right")
    ]
    alignment = [
        {
            "tool_ref": tools[side].tool_ref,
            "target_pose": dict(item["target_pose"]),
        }
        for side, item in zip(("left", "right"), pregrasp, strict=True)
    ]
    if secondary_side is not None and secondary_front_transfer is not None:
        next(
            item
            for item in alignment
            if item["tool_ref"] == tools[secondary_side].tool_ref
        )["target_pose"] = secondary_front_transfer
        # 第二侧在凹槽中心线外侧下降到插钩高度。首侧末端在整段路径中
        # 保持固定；SDK仍逐点检查真实箱沿、邻箱和Robot碰撞，这里只去掉
        # 绕前角造成的大范围扫掠，不放宽任何接触或到位条件。
        next(
            item
            for item in pregrasp
            if item["tool_ref"] == tools[secondary_side].tool_ref
        )["target_pose"] = at(
            secondary_side,
            radial_offset=approach_clearance_by_side[secondary_side],
            extra_z=-engagement_travel,
            lane_offset_override=front_lane_offset,
        )
    approach = [
        {
            "tool_ref": tools[side].tool_ref,
            "target_pose": at(
                side,
                radial_offset=approach_clearance_by_side[side],
                extra_z=-engagement_travel,
            ),
        }
        for side in ("left", "right")
    ]
    insert = [
        {
            "tool_ref": tools[side].tool_ref,
            # 工具目标使用固定下钩的load frame，不是腕部、夹具外表面或
            # 钩脚几何中心。
            # “插入深度”从箱体外轮廓向凹腔内部量取。旧实现使用正偏移，
            # 末端虽然精确到位，下钩却仍停在箱体外侧，夹紧只能空行程。
            # approach 先在槽口下方留出少量垂直间隙，避免长距离接近时扫过
            # 箱沿；insert 的末端则回到凹槽插入高度，使钩体在水平插入的同时
            # 完成一次很短的向上就位。真实 Runtime 必须随后报告凹槽上沿接触，
            # 仅碰到凹槽后壁或箱体外表面不能视为插钩成功。
            "target_pose": at(
                side,
                radial_offset=-insertion,
                extra_z=-engagement_travel,
                lane_offset_override=(
                    secondary_lane_offset_override
                    if side == secondary_side
                    else None
                ),
            ),
        }
        for side in ("left", "right")
    ]
    seat = [
        {
            "tool_ref": tools[side].tool_ref,
            "target_pose": at(
                side,
                radial_offset=-insertion,
                extra_z=0.0,
                lane_offset_override=(
                    secondary_lane_offset_override
                    if side == secondary_side
                    else None
                ),
            ),
        }
        for side in ("left", "right")
    ]
    opening = [
        {
            "tool_ref": tools[side].tool_ref,
            # 预抓取开度由夹具 Provider 根据设备行程给出。Skill 只执行候选，
            # 不能把某个 MuJoCo 夹具的几何常量写进通用控制流程。
            "target_position_m": min(pregrasp_opening, tools[side].travel_m),
            "maximum_force_n": tools[side].normal_force_n,
            "hold": False,
        }
        for side in ("left", "right")
    ]
    clamp = [
        {
            "tool_ref": tools[side].tool_ref,
            # 闭合位置取决于具体夹具和周转箱抓取凹槽的几何关系。通用 Skill 只执行
            # Provider 返回的候选，SDK 只负责约束工具行程；若在这里固定 15mm，
            # 真实夹具会在接触前停止。配置值仍受 Profile 声明的实际行程限制。
            "target_position_m": min(clamp_target_position, tools[side].travel_m),
            # 夹紧设定来自工具Profile的额定连续力；maximum_force_n是驱动安全
            # 上限，不是应长期维持的目标力。外拉工艺若需要不同设定，应由明确
            # 的工具控制契约表达，不能把峰值上限偷偷解释成夹紧目标。
            "maximum_force_n": tools[side].normal_force_n,
            "hold": True,
        }
        for side in ("left", "right")
    ]
    pull_side = "left" if strategy == "left_extract_first" else "right"
    pull_path = []
    pull_direction_world = None
    pull_distance_m = 0.0
    if strategy != "direct_bilateral":
        # 外拉距离不是某个箱型的固定轨迹参数。Provider根据当前箱体尺寸、
        # 同层邻箱表面间隙和夹具实际行程计算打开另一侧接近通道，再受Profile
        # 声明的设备最大外拉边界约束。Runtime、SDK和Skill均不理解该业务几何。
        pull_sign = side_sign[pull_side]
        # 根据实时邻箱间隙计算一次主拉取所需距离。工具已经通过下钩和
        # 上夹片形成受控接触，不能再用insertion/4把厘米级缺口拆成大量
        # 毫米级动作；主拉后重新观测，最多再做一次基于新缺口的细调。
        maximum_distance = float(
            settings.get(
                "maximum_extraction_distance_m",
                max(clearance, tool_access_envelope),
            )
        )
        # 首侧接合前只需要完成一次短拉并立即重观测。接合后，停止条件是
        # 第二侧整条Robot路径真实可达，而不仅是夹具尖端刚好塞得进表面间隙。
        # approach_clearance描述该工具/机械臂组合的接近扫掠空间，并受设备最大
        # 外拉边界限制；每个短步后SDK碰撞预检都可以更早结束外拉。
        # 首侧已经接合后，继续外拉不仅要补足钩脚入口，还要让第二侧完整Robot
        # 路径通过碰撞预检。名义净空满足后若腕部/前臂仍撞到同一邻箱，下一轮
        # 允许再扩大一个厘米级通道；路径一旦可达立即停止，不能按固定次数盲拉。
        required_gap = (
            secondary_path_gap if engaged_side is not None else initial_extraction_gap
        )
        if (
            engaged_side is not None
            and math.isfinite(available_gap)
            and available_gap < secondary_path_gap
        ):
            # 首侧刚在凹槽中形成受控接触时，只补到“钩脚勉强能进”
            # 会把一次主拉拆成毫米级动作，右腕和前臂仍没有完整通道。
            # 因此第一次按实时邻箱缺口形成接近工作净空的连续外拉；
            # 动作后立即重新观测。入口已足够时不再追距离尾差，而是交给
            # SDK完整IK/碰撞预检决定是否还需要补拉。
            required_gap = initial_extraction_gap
        required_distance = 0.0
        for item in scene_objects:
            if str(getattr(item, "source_id", "")) == str(data["object_ref"]):
                continue
            if str(getattr(item, "category", "")) != "tote":
                continue
            other_pose = getattr(item, "pose", None)
            other_extent_raw = getattr(item, "extent", None)
            if other_pose is None or other_extent_raw is None:
                continue
            other_extent = tuple(float(value) for value in other_extent_raw)
            delta = tuple(float(other_pose.position[i]) - center[i] for i in range(3))
            if abs(delta[2]) >= (extent[2] + other_extent[2]) / 2.0 - 0.01:
                continue
            other_lane_span = _extent_span_along_world_axis(
                other_extent, other_pose.quaternion_xyzw, lane_axis_world
            )
            if (
                abs(_dot(lane_axis_world, delta))
                >= (target_lane_span + other_lane_span) / 2.0
            ):
                continue
            axis_projection = _dot(groove_axis_world, delta)
            # 只计算外拉反方向的邻箱；外拉方向的通道是否可用已由候选排序
            # 判断，后续每个Action仍会做实时IK和碰撞检查。
            if axis_projection * pull_sign >= 0.0:
                continue
            other_span = _extent_span_along_world_axis(
                other_extent, other_pose.quaternion_xyzw, groove_axis_world
            )
            surface_gap = abs(axis_projection) - (target_axis_span + other_span) / 2.0
            # 工具参考系靠近目标箱表面，因此外拉只需让同层同排通道
            # 容纳工具朝向邻箱一侧的真实径向包络。其他排、其他层的箱体不
            # 属于这条进入通道，不能造成永不结束的补拉。
            required_distance = max(
                required_distance,
                required_gap - surface_gap,
            )
        # 单次外拉上限属于夹具与凹槽接口的工艺边界，由Provider配置；
        # 实际需要量仍来自本轮SceneSnapshot中的箱体间隙。每步结束后Skill会
        # 重新观测箱体，不能把命令位移累计成物体已经移动的事实。
        configured_maximum_step = float(
            settings.get("maximum_extraction_step_m", maximum_distance)
        )
        if (
            engaged_side is not None
            and secondary_preflight_mode == "path"
            and math.isfinite(available_gap)
            and available_gap < maximum_distance
        ):
            # 这一条pull_path是路径预检失败时才会使用的备用动作。一次补拉至少
            # 使用完整工具包络，且最多使用接近净空的一半，避免重新退化成数毫米
            # 的试探动作；真正上限仍由设备Profile约束。若预检已经通过，下面的
            # reachability会清空该路径，Skill不会多拉一次。
            adaptive_path_increment = min(
                configured_maximum_step,
                max(tool_access_envelope, clearance / 2.0),
                maximum_distance - available_gap,
            )
            required_distance = max(required_distance, adaptive_path_increment)
        # Profile只限制一次连续外拉的机械上限；实际距离仍由实时通道缺口
        # 决定。此前再用 insertion/4 截断会把一次约12mm的动作拆成多个
        # 2～3mm动作，每次重新规划都会重复克服静摩擦并消耗钩爪重叠。
        maximum_step = configured_maximum_step
        remaining_extraction_distance = max(
            0.0, min(required_distance, maximum_distance)
        )
        pull_distance = min(remaining_extraction_distance, maximum_step)
        if engaged_side is None or pull_side == engaged_side:
            if pull_distance > 1e-4:
                if engaged_side is not None:
                    # 第一侧不能为了恢复理想几何而打开压紧片：真实测试证明，
                    # 上层箱仍受下层摩擦约束时，一旦卸载压紧片，约1N的钩接触
                    # 无法在重新就位运动中维持。下一小步因此从实时末端Pose
                    # 连续执行；理想seat只提供工具插入方向，不能再作为位置目标。
                    # 本次位移和当前箱体Pose都来自实时Robot/Scene，不按箱型、
                    # 层高或MuJoCo帧数维护固定动作模板。
                    engaged_seat = next(
                        item["target_pose"]
                        for item in seat
                        if item["tool_ref"] == tools[engaged_side].tool_ref
                    )
                    engaged_current = robot_state.end_effectors.get(engaged_side)
                    if engaged_current is None:
                        raise ValueError(f"Robot状态缺少{engaged_side}末端位姿")
                    insertion_axis = _rotate_vector(
                        engaged_seat["orientation_xyzw"], (1.0, 0.0, 0.0)
                    )
                    pull_direction_world = [-value for value in insertion_axis]
                    pull_pose = {
                        "frame_id": engaged_current.frame_id,
                        "position_m": [
                            float(engaged_current.position[i])
                            - insertion_axis[i] * pull_distance
                            for i in range(3)
                        ],
                        "orientation_xyzw": list(engaged_current.quaternion_xyzw),
                        "observed_at": engaged_seat.get("observed_at"),
                        "revision": data["target_revision"],
                    }
                    # 单侧钩入后的补拉沿当前抓取凹槽水平执行。实物Gate证明，
                    # 单侧浅抬不会卸载整箱，反而会让工具相对箱体向上滑并缩短
                    # 钩爪重叠。当前箱体Pose仍在每步后重新观测，第二侧何时进入
                    # 则由真实通道宽度和完整Robot路径共同决定。
                    pull_pose["position_m"][2] = float(engaged_current.position[2])
                else:
                    pull_direction_world = [
                        groove_axis_world[index] * pull_sign for index in range(3)
                    ]
                    pull_pose = at(
                        pull_side,
                        radial_offset=-insertion + pull_distance,
                        extra_z=0.0,
                    )
                pull_distance_m = pull_distance
                pull_path = [
                    {
                        "tool_ref": tools[pull_side].tool_ref,
                        "target_pose": pull_pose,
                    }
                ]
    result = {
        "candidate_id": f"{strategy}-{index}",
        "object_ref": data["object_ref"],
        "observation_revision": data["target_revision"],
        "planned_object_pose": pose,
        "strategy": strategy,
        "tool_targets": tool_targets,
        "clearance_poses": clearance_poses,
        "approach_poses": approach,
        "hook_insert_poses": insert,
        "hook_seat_poses": seat,
        "opening_setpoints": opening,
        "clamp_setpoints": clamp,
        "transfer_poses": transfer,
        "alignment_poses": alignment,
        "pregrasp_poses": pregrasp,
        "pull_path": pull_path,
        "pull_direction_world": pull_direction_world,
        "pull_distance_m": pull_distance_m,
        "clearance_m": clearance,
        "score": max(0.0, 1.0 - (index - 1) * 0.1),
        # 中空箱体只在GraspPlanning的碰撞快照中展开为普通box组合。
        # 这不是Skill输入，也不进入SDK领域模型；候选返回前会被移除。
        "_target_collision_geometry": {
            "pose": pose,
            "extent_m": extent,
            "groove_axis_local": list(groove_axis_local),
            "groove_width_m": float(settings.get("groove_width_m", 0.0)),
            "groove_height_m": float(settings.get("groove_height_m", 0.0)),
            "groove_depth_m": groove_depth,
            "groove_top_offset_m": groove_top_offset,
            "wall_thickness_m": float(settings.get("wall_thickness_m", 0.0)),
        },
    }
    if engaged_side is not None:
        result["_secondary_preflight_mode"] = secondary_preflight_mode
        result["_secondary_obstacle_ids"] = secondary_obstacle_ids
        result["_secondary_resume_tolerance_m"] = max(
            engagement_travel,
            motion_position_tolerance_m * 2.0,
        )
    return result


def _normalized_axis(raw: Any) -> tuple[float, float, float]:
    values = tuple(float(value) for value in raw)
    if len(values) != 3:
        raise ValueError("groove_axis_local 必须是三个数")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-8:
        raise ValueError("groove_axis_local 不能是零向量")
    return tuple(value / norm for value in values)


def _rotate_vector(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    """用公共 xyzw 四元数旋转方向向量，不引入数值计算依赖。"""

    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _extent_span_along_world_axis(
    extent: Sequence[float],
    orientation_xyzw: Sequence[float],
    world_axis: Sequence[float],
) -> float:
    """把物体局部extent投影到指定世界方向。"""

    qx, qy, qz, qw = (float(value) for value in orientation_xyzw)
    local_axis = _rotate_vector((-qx, -qy, -qz, qw), world_axis)
    return sum(
        abs(float(local_axis[index])) * float(extent[index])
        for index in range(3)
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _inward_tool_orientation(
    groove_axis_world: Sequence[float], side_sign: float
) -> list[float]:
    """让工具局部 +X 沿水平抓取凹槽轴指向箱体中心。"""

    inward_x = -side_sign * float(groove_axis_world[0])
    inward_y = -side_sign * float(groove_axis_world[1])
    yaw = math.atan2(inward_y, inward_x)
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
