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

"""各 Ability 共用的无设备状态工具。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from semantic_robot_sdk_core import (
    EnvironmentCollisionObject,
    EnvironmentCollisionSet,
    Pose,
)
from semantic_robot_sdk_r1pro import R1ProSDK


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def invocation_id(task_name: str, digest: str) -> str:
    return f"inv-{task_name.lower()}-{digest[:20]}"


def pose(value: Mapping[str, Any]) -> Pose:
    return Pose(
        frame_id=str(value["frame_id"]),
        position=tuple(float(item) for item in value["position_m"]),
        quaternion_xyzw=tuple(float(item) for item in value["orientation_xyzw"]),
    )


def pose_dict(value: Pose, *, revision: str) -> dict[str, Any]:
    return {
        "frame_id": value.frame_id,
        "position_m": list(value.position),
        "orientation_xyzw": list(value.quaternion_xyzw),
        "observed_at": now_iso(),
        "revision": revision,
    }


def fixture_pose(revision: str, *, x: float = 1.0, z: float = 0.25) -> dict[str, Any]:
    return {
        "frame_id": "world",
        "position_m": [x, 0.0, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "observed_at": now_iso(),
        "revision": revision,
    }


def observation(
    kind: str,
    source: str,
    subject_ref: str | None,
    value: Mapping[str, Any],
    *,
    revision: str | None = None,
    frame_id: str | None = None,
    confidence: float | None = None,
    artifact_refs: list[str] | None = None,
) -> dict[str, Any]:
    digest = request_digest({"kind": kind, "subject_ref": subject_ref, "value": value})[
        :20
    ]
    return {
        "id": f"obs-{digest}",
        "kind": kind,
        "schema_version": 2,
        "subject_ref": subject_ref,
        "source": source,
        "observed_at": now_iso(),
        "revision": revision,
        "frame_id": frame_id,
        "confidence": confidence,
        "value": dict(value),
        "data_ref": None,
        "artifact_refs": list(artifact_refs or ()),
        "evidence_refs": [f"evidence://ability/{digest}"],
    }


def component_name(sdk: R1ProSDK, reference: str | None, role: str) -> str:
    """把 tool_ref 解析为 Profile 中的逻辑侧；不根据厂商关节名猜测。"""

    raw = reference or ""
    profile = sdk.state.capabilities()
    for tool in profile.tools:
        if raw in {tool.tool_ref, tool.side}:
            return tool.side
    if raw.rstrip("/").rsplit("/", 1)[-1] == "main":
        settings = sdk.deployment.ability_settings(role)
        return str(settings.get("default_tool_side", "left"))
    raise ValueError(f"Robot Profile 不包含工具：{raw}")


def scene_collision_environment(
    sdk: R1ProSDK,
    *,
    exclude_source_ids: tuple[str, ...] = (),
    allowed_contact_end_effectors_by_source: Mapping[str, Sequence[str]] | None = None,
) -> EnvironmentCollisionSet | None:
    """把 Backend 的实时场景对象转换为一次运动规划使用的碰撞快照。

    SceneSnapshot 是 Runtime/真机 Provider 给出的当前几何事实，不是
    Semantic Map。Ability 根据动作语义决定是否排除正在接触或搬运的目标，
    Robot SDK 只负责对引擎无关的 box/sphere 做轨迹碰撞检查。没有场景来源的
    Backend（例如最小 Fake 测试）仍可依赖设备自身安全能力，不伪造障碍物。
    """

    try:
        snapshot = sdk.state.scene_snapshot()
    except (AttributeError, NotImplementedError, OSError, RuntimeError, ValueError):
        return None
    excluded = set(exclude_source_ids)
    allowed_contacts = allowed_contact_end_effectors_by_source or {}
    objects = []
    for item in snapshot.objects:
        if item.source_id in excluded or item.extent is None:
            continue
        extent = tuple(float(value) for value in item.extent)
        if any(value <= 0 for value in extent):
            continue
        objects.append(
            EnvironmentCollisionObject(
                source_id=item.source_id,
                shape="box",
                pose=item.pose,
                size_xyz=extent,
                allowed_contact_end_effectors=tuple(
                    allowed_contacts.get(item.source_id, ())
                ),
            )
        )
    return EnvironmentCollisionSet(
        frame_id=snapshot.coordinate_frame,
        objects=objects,
    )
