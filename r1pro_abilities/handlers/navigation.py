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

"""Navigation Ability：路线规划、执行与独立到达验证。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping

from semantic_robot_sdk_core import MotionPlan, PlanningError

from ..ability_utils import component_name, now_iso, observation, pose, pose_dict
from ..execution import ExecutionRecord, ExecutionStatus
from ..tool_load import (
    load_failure_requires_stop,
    parse_load_unstable_since,
    read_tool_load,
)
from .base import AbilityHandler


class NavigationHandler(AbilityHandler):
    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        if task_name == "PlanRoute":
            return self._plan_route(data, invocation_id, digest)
        if task_name == "FollowRoute":
            if data.get("reason"):
                return self._hold(data, invocation_id, digest)
            return self._follow_route(data, invocation_id, digest)
        if task_name == "VerifyArrival":
            return self._verify_arrival(data, invocation_id, digest)
        raise ValueError(f"Navigation 不支持 Task: {task_name}")

    def after_refresh(self, record: ExecutionRecord) -> ExecutionRecord:
        if record.task_name != "FollowRoute" or not record.request.get(
            "carrying_object"
        ):
            return record
        carrying = record.request["carrying_object"]
        tool_refs = [str(value) for value in carrying.get("tool_refs", [])]
        state, load = read_tool_load(self.sdk, tool_refs)
        stable = load.safe
        tool_states = {
            ref: (value.model_dump(mode="json") if value is not None else None)
            for ref, value in load.tool_states.items()
        }
        obs = observation(
            "navigation.carrying_load",
            f"robot-sdk://{self.sdk.robot_id}/state",
            str(carrying["object_ref"]),
            {
                "stable": stable,
                "slipping": load.slipping,
                "overloaded": load.overloaded,
                "sensor_fault": load.sensor_fault,
                "reasons": list(load.reasons),
                # 这里保留真机同样能够提供的实时传感数值，供执行页和恢复
                # 逻辑区分真实滑移与短时振动；不暴露MuJoCo帧数或Runtime
                # 内部稳定结论，也不在Navigation里复制一套阈值判断。
                "tool_states": tool_states,
            },
            revision=f"carry-{state.generation}-{state.observed_at.isoformat()}",
            confidence=1.0,
        )
        unstable_since = parse_load_unstable_since(
            record.result.get("_load_unstable_since")
        )
        result: dict[str, Any] = {}
        if stable:
            # 这只是 Ability 的短时安全判定状态，不是 Runtime 的业务状态。
            # 一旦原始传感状态恢复，立即清除先前的异常观察窗口。
            result["_load_unstable_since"] = None
        elif unstable_since is None:
            unstable_since = datetime.now(timezone.utc)
            result["_load_unstable_since"] = unstable_since.isoformat()
        record = self.executions.update_observations(
            record.invocation_id, (obs,), result=result
        )
        if (
            not stable
            and record.status
            in {
                ExecutionStatus.ACCEPTED,
                ExecutionStatus.RUNNING,
            }
            and load_failure_requires_stop(load, unstable_since)
        ):
            # 传感故障或超力能由一次采样直接证明，必须立即停止。接触/相对
            # 运动异常则要持续超过真实时间窗口：底盘刚启动时一次速度尖峰不
            # 等于箱体已经滑落。这个窗口只存在于 Ability，不依赖 MuJoCo
            # timestep，也不向 Robot Skill 暴露“物理帧”计数。
            return self.executions.stop(
                self.sdk, record.invocation_id, "carrying_load_unstable"
            )
        return record

    def _plan_route(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        carrying = data.get("carrying_object")
        plan = None
        selected_goal = None
        last_error: PlanningError | None = None
        for goal in self._route_goal_candidates(data):
            try:
                plan = self.sdk.base.plan_route(
                    goal,
                    maximum_speed_mps=float(data["maximum_speed_mps"]),
                    minimum_clearance_m=float(data["minimum_clearance_m"]),
                    carrying_object_ref=(
                        str(carrying["object_ref"]) if carrying else None
                    ),
                    carrying_object_pose=(
                        pose(carrying["object_pose"]) if carrying else None
                    ),
                    carrying_object_extent_m=(
                        tuple(float(value) for value in carrying["object_size_m"])
                        if carrying
                        else None
                    ),
                )
            except PlanningError as exc:
                last_error = exc
                continue
            selected_goal = goal
            break
        if plan is None or selected_goal is None:
            # 路线规划只读取实时状态和 SceneSnapshot；PlanningError 明确发生在
            # 下发底盘命令之前。把它保存为 failed 后 Robot Agent 可以调整目标，
            # 网络断连等无法确认是否启动的错误仍由 Pilot 保留为 interrupted。
            return self.failed(
                "PlanRoute",
                data,
                invocation_id,
                digest,
                "PLANNING_FAILED",
                str(last_error or "没有可用的导航工位"),
            )
        # 相同目标在 Scene reset 前后可能产生完全相同的请求摘要，但底层
        # MotionPlan 只能用于生成它的 Runtime generation。把 generation
        # 编入路线身份，可避免 Execution Store 按旧 route_ref 找到 reset 前
        # 的计划；这里描述的是 Runtime 实时状态，不是 Semantic Map 版本。
        route_ref = f"route-g{plan.generation}-{digest[:16]}"
        route_revision = f"g{plan.generation}-{digest[:12]}"
        resolved_target = dict(data["target"])
        resolved_pose = {
            "frame_id": selected_goal.frame_id,
            "position_m": list(selected_goal.position),
            "orientation_xyzw": list(selected_goal.quaternion_xyzw),
        }
        resolved_pose["position_m"][:2] = [plan.goal["x"], plan.goal["y"]]
        requested_pose = data["target"].get("pose")
        if isinstance(requested_pose, Mapping):
            for key in ("observed_at", "revision"):
                if requested_pose.get(key) is not None:
                    resolved_pose[key] = requested_pose[key]
        resolved_target["pose"] = resolved_pose
        return self.succeeded(
            "PlanRoute",
            data,
            invocation_id,
            digest,
            {
                "route_ref": route_ref,
                "route_revision": route_revision,
                "resolved_target": resolved_target,
                "_motion_plan": plan.model_dump(mode="json"),
            },
        )

    def _route_goal_candidates(self, data: Mapping[str, Any]) -> tuple[Any, ...]:
        """把业务对象/槽位解析成当前场景中的Robot基座工位候选。"""

        requested = pose(data["target"]["pose"])
        if data.get("navigation_purpose") == "transit":
            return (requested,)
        try:
            snapshot = self.sdk.state.scene_snapshot()
            state = self.sdk.state.snapshot()
        except (RuntimeError, OSError):
            return (requested,)
        if state.base_pose is None:
            return (requested,)

        target_ref = str(data["target"]["target_ref"])
        leaf = target_ref.rstrip("/").rsplit("/", 1)[-1]
        targets = [
            item
            for item in (*snapshot.objects, *snapshot.regions)
            if item.source_id in {target_ref, leaf}
        ]
        if len(targets) != 1 or targets[0].extent is None:
            return (requested,)
        target = targets[0]

        support_ref = str(
            getattr(target, "properties", {}).get("support_surface_ref", "")
        )
        supports = [
            item
            for item in snapshot.objects
            if item.extent is not None
            and (
                (support_ref and item.source_id == support_ref)
                or (
                    not support_ref
                    and item.category == "pallet"
                    and abs(target.pose.position[0] - item.pose.position[0])
                    <= item.extent[0] / 2
                    and abs(target.pose.position[1] - item.pose.position[1])
                    <= item.extent[1] / 2
                    and item.pose.position[2] <= target.pose.position[2]
                )
            )
        ]
        if len(supports) != 1:
            return (requested,)
        support = supports[0]

        options = self.sdk.deployment.robot.sdk.options
        work_distance = float(options.get("manipulation_work_distance_m", 0.0))
        resolution = float(options.get("navigation_resolution_m", 0.05))
        capabilities = getattr(self.sdk.navigation_provider, "capabilities", None)
        footprint = getattr(capabilities, "base_footprint_radius_m", None)
        if work_distance <= 0 or footprint is None:
            return (requested,)
        margin = max(float(footprint), float(data["minimum_clearance_m"])) + resolution

        tx, ty, tz = target.pose.position
        target_x, target_y, _ = target.extent
        sx, sy, _ = support.pose.position
        support_x, support_y, _ = support.extent
        support_min_x, support_max_x = sx - support_x / 2, sx + support_x / 2
        support_min_y, support_max_y = sy - support_y / 2, sy + support_y / 2

        # 先保持周转箱宽面作业。每个候选同时满足目标表面工作距离和支撑托盘
        # 外侧净空；前后排因此会自然选择南/北两侧，而不是让模型手算底盘Pose。
        raw_candidates = [
            (
                (tx, min(ty - target_y / 2 - work_distance, support_min_y - margin)),
                0.0 if target_x >= target_y else resolution,
                abs(
                    (ty - target_y / 2)
                    - min(ty - target_y / 2 - work_distance, support_min_y - margin)
                    - work_distance
                ),
            ),
            (
                (tx, max(ty + target_y / 2 + work_distance, support_max_y + margin)),
                0.0 if target_x >= target_y else resolution,
                abs(
                    max(ty + target_y / 2 + work_distance, support_max_y + margin)
                    - (ty + target_y / 2)
                    - work_distance
                ),
            ),
            (
                (min(tx - target_x / 2 - work_distance, support_min_x - margin), ty),
                0.0 if target_y > target_x else resolution,
                abs(
                    (tx - target_x / 2)
                    - min(tx - target_x / 2 - work_distance, support_min_x - margin)
                    - work_distance
                ),
            ),
            (
                (max(tx + target_x / 2 + work_distance, support_max_x + margin), ty),
                0.0 if target_y > target_x else resolution,
                abs(
                    max(tx + target_x / 2 + work_distance, support_max_x + margin)
                    - (tx + target_x / 2)
                    - work_distance
                ),
            ),
        ]
        current = state.base_pose.position[:2]
        qx, qy, qz, qw = state.base_pose.quaternion_xyzw
        current_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        carrying = data.get("carrying_object")
        desired_carry_yaw = current_yaw
        if carrying:
            object_orientation = (carrying.get("object_pose") or {}).get(
                "orientation_xyzw"
            )
            if object_orientation and len(object_orientation) == 4:
                oqx, oqy, oqz, oqw = (float(value) for value in object_orientation)
                object_yaw = math.atan2(
                    2.0 * (oqw * oqz + oqx * oqy),
                    1.0 - 2.0 * (oqy * oqy + oqz * oqz),
                )
                tqx, tqy, tqz, tqw = target.pose.quaternion_xyzw
                target_yaw = math.atan2(
                    2.0 * (tqw * tqz + tqx * tqy),
                    1.0 - 2.0 * (tqy * tqy + tqz * tqz),
                )
                # 箱体与基座在携物期间保持相对方向。目标工位应补偿箱体当前
                # 朝向与槽位朝向之差，不能只按离基座最近的一侧停靠，否则会
                # 把箱体旋转 90 度，迫使 place-object 承担不可达的大角度整姿。
                desired_carry_yaw = current_yaw + target_yaw - object_yaw

        def candidate_rank(item):
            (x, y), face_penalty, work_error = item
            candidate_yaw = math.atan2(ty - y, tx - x)
            yaw_delta = candidate_yaw - desired_carry_yaw
            turn_error = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
            if carrying:
                # 携物放置先选与箱体宽面相符且能对齐目标朝向的工位；所有
                # 候选都已经满足托盘外侧净空，因此无需用几厘米的距离优势
                # 换取 90 度携物转向。距离只用于同方向候选的最终排序。
                return (
                    round(face_penalty, 6),
                    turn_error,
                    round(work_error, 6),
                    math.dist(current, (x, y)),
                )
            # 抓取工位首先选择箱体宽面。工作距离从0.55m调近到0.48m后，
            # 托盘外侧净空会让宽面候选比理论值多约10cm；若把这项误差与
            # 朝向罚分相加，数值更“精确”的窄面会反而胜出，双臂随后只能
            # 从箱体短边接近并撞上目标箱体。R1 Pro 双臂抓取还需要稳定使用
            # 候选表中的标准负Y作业面；南北两面几何等价时不能随Robot当前位置
            # 翻转180度，否则同一个箱体会落入另一套接近构型并失去可达候选。
            # Python稳定排序会保留上方候选定义的标准面顺序。
            return (
                round(face_penalty, 6),
                round(work_error, 6),
            )

        raw_candidates.sort(
            key=candidate_rank
        )

        goals = []
        seen = set()
        for (x, y), _face_penalty, _work_error in raw_candidates:
            key = (round(x, 6), round(y, 6))
            if key in seen:
                continue
            seen.add(key)
            yaw = math.atan2(ty - y, tx - x)
            goals.append(
                type(requested)(
                    frame_id=target.pose.frame_id,
                    position=(x, y, state.base_pose.position[2]),
                    quaternion_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                )
            )
        requested_key = (
            round(requested.position[0], 6),
            round(requested.position[1], 6),
        )
        if requested_key not in seen:
            goals.append(requested)
        return tuple(goals)

    def _follow_route(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        route = self.executions.find_result_value("route_ref", data["route_ref"])
        if route is None or not route.result.get("_motion_plan"):
            return self.failed(
                "FollowRoute",
                data,
                invocation_id,
                digest,
                "ROUTE_UNKNOWN",
                "路线引用不存在或已经失效，必须重新规划",
            )
        plan = MotionPlan.model_validate(route.result["_motion_plan"])
        command = self.sdk.base.follow_route(
            plan,
            command_id=f"{invocation_id}:command",
        )
        return self.register_command(
            command,
            "FollowRoute",
            data,
            invocation_id,
            digest,
            resource="base",
            result={
                "final_pose_ref": f"robot-state://{self.sdk.robot_id}/base/{data['route_ref']}",
                "distance_to_target_m": 0.0,
            },
        )

    def _verify_arrival(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        state = self.sdk.state.snapshot()
        if state.base_pose is None:
            return self.failed(
                "VerifyArrival",
                data,
                invocation_id,
                digest,
                "BASE_POSE_UNAVAILABLE",
                "Robot SDK 没有返回底盘位姿",
            )
        target = pose(data["target"]["pose"])
        distance = math.dist(state.base_pose.position, target.position)
        carrying = data.get("carrying_object")
        observations: tuple[Mapping[str, Any], ...] = ()
        if carrying is not None:
            refs = [str(value) for value in carrying.get("tool_refs", [])]
            state, load = read_tool_load(self.sdk, refs)
            held = load.safe
            revision = f"arrival-{digest[:16]}"
            refreshed = None
            if held:
                refreshed = dict(carrying)
                verified_at = now_iso()
                tool_poses: dict[str, dict[str, Any]] = {}
                for tool_ref in refs:
                    side = component_name(self.sdk, tool_ref, "robot_state")
                    if side not in state.end_effectors:
                        return self.failed(
                            "VerifyArrival",
                            data,
                            invocation_id,
                            digest,
                            "END_EFFECTOR_POSE_UNAVAILABLE",
                            f"Robot SDK 未返回 {tool_ref} 的末端位姿",
                        )
                    tool_poses[tool_ref] = pose_dict(
                        state.end_effectors[side], revision=revision
                    )

                object_pose = None
                object_size_m = None
                if self.sdk.deployment.robot.backend == "mujoco":
                    scene = self.sdk.state.scene_snapshot()
                    object_ref = str(carrying["object_ref"])
                    leaf = object_ref.rstrip("/").rsplit("/", 1)[-1]
                    matches = [
                        item
                        for item in scene.objects
                        if item.source_id in {object_ref, leaf}
                    ]
                    if len(matches) != 1 or matches[0].extent is None:
                        return self.failed(
                            "VerifyArrival",
                            data,
                            invocation_id,
                            digest,
                            "OBJECT_GEOMETRY_UNAVAILABLE",
                            "MuJoCo 场景未返回唯一物体位姿和尺寸",
                        )
                    object_pose = pose_dict(matches[0].pose, revision=revision)
                    object_size_m = list(matches[0].extent)
                else:
                    # 真机若没有独立物体跟踪源，使用导航后双工具实时位姿的
                    # 几何中心更新持物估计，不能继续复制导航前的 object_pose。
                    # 后续 Place Stage 仍会重新读取 RobotState 并实时观察槽位。
                    first_pose = tool_poses[refs[0]]
                    positions = [tool_poses[ref]["position_m"] for ref in refs]
                    object_pose = dict(first_pose)
                    object_pose["position_m"] = [
                        sum(position[axis] for position in positions) / len(positions)
                        for axis in range(3)
                    ]
                    object_size_m = list(refreshed["object_size_m"])
                refreshed.update(
                    {
                        "object_pose": object_pose,
                        "object_size_m": object_size_m,
                        "tool_poses": tool_poses,
                        "grasp_pose": tool_poses[refs[0]],
                        "verified_at": verified_at,
                        "robot_state_revision": revision,
                        "evidence_refs": list(
                            dict.fromkeys(
                                [
                                    *refreshed.get("evidence_refs", ()),
                                    f"evidence://{self.sdk.robot_id}/arrival-held/{revision}",
                                ]
                            )
                        ),
                    }
                )
            carrying = refreshed
            observations = (
                observation(
                    "manipulation.held_object",
                    f"robot-sdk://{self.sdk.robot_id}/state",
                    str(data["carrying_object"]["object_ref"]),
                    {
                        "state": refreshed,
                        "held": held,
                        "stable": held,
                        "slipping": load.slipping,
                        "overloaded": load.overloaded,
                        "sensor_fault": load.sensor_fault,
                        "reasons": list(load.reasons),
                    },
                    revision=revision,
                    frame_id=state.base_pose.frame_id,
                    confidence=1.0,
                ),
            )
        arrived = distance <= float(data["arrival_radius_m"])
        carrying_ok = data.get("carrying_object") is None or carrying is not None
        result = {
            "verdict": "achieved" if arrived and carrying_ok else "not_achieved",
            "final_pose_ref": f"robot-state://{self.sdk.robot_id}/base/latest",
            "distance_to_target_m": distance,
            "carrying_object": carrying,
        }
        return self.succeeded(
            "VerifyArrival", data, invocation_id, digest, result, observations
        )

    def _hold(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        state = self.sdk.safety.hold()
        confirmed = bool(state.in_hold)
        return self.executions.register_terminal(
            "FollowRoute",
            digest,
            ExecutionStatus.SUCCEEDED if confirmed else ExecutionStatus.INTERRUPTED,
            {
                "safe": confirmed,
                "physical_state": "hold" if confirmed else "unknown",
                "stop_evidence": {"holding": confirmed, "reason": data["reason"]},
            },
            invocation_id,
            error=None
            if confirmed
            else {"code": "HOLD_UNCONFIRMED", "message": "未取得底盘 hold 证据"},
            request=data,
        )
