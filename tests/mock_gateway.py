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

"""完整 Mock 验收使用的 AbilityFramework 兼容进程。

该入口只用于开发与测试。它继续使用 ability_py.task_manager、正式 Ability Service
和 Fake Robot SDK；场景故障注入被限制在本文件，生产 Ability 不包含演示分支。
stdout 只输出 JSON-RPC，诊断信息必须写入 stderr。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ability_py import TaskInterface
from ability_py.task_manager import TaskManager, TaskStatus
from semantic_robot_sdk_core import load_robot_deployment
from semantic_robot_sdk_r1pro import R1ProSDK

from r1pro_abilities.models import ModelProfileRegistry
from r1pro_abilities.service import TASKS, AbilityRole, R1ProAbilityService


_INSTANCE_NAMES = {
    role: f"fake-r1-{role.value.replace('_', '-')}" for role in AbilityRole
}
_TASK_ROLES = {
    task_name: role for role, task_names in TASKS.items() for task_name in task_names
}


class _DispatchTask(TaskInterface):
    def __init__(self, gateway: "MockAbilityGateway", task_name: str) -> None:
        self._gateway = gateway
        self._task_name = task_name

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return self._gateway.execute(self._task_name, input_data)


class MockAbilityGateway:
    """用一个共享 Fake SDK Session 托管七个逻辑 Ability 接口。"""

    def __init__(self, profile_path: Path, model_path: Path, store_dir: Path) -> None:
        self.sdk = R1ProSDK.from_deployment(load_robot_deployment(profile_path))
        self.models = ModelProfileRegistry.load(model_path)
        store_dir.mkdir(parents=True, exist_ok=True)
        self.services = {
            role: R1ProAbilityService(
                role,
                self.sdk,
                self.models
                if role
                in {
                    AbilityRole.OBJECT_PERCEPTION,
                    AbilityRole.GRASP_PLANNING,
                }
                else None,
                store_dir / f"{role.value}.sqlite",
            )
            for role in AbilityRole
        }
        self.manager = TaskManager()
        self.indexes: dict[tuple[str, str], int] = {}
        self.invocation_roles: dict[str, AbilityRole] = {}
        self.synthetic: dict[str, dict[str, Any]] = {}
        self.counts: Counter[str] = Counter()
        handlers: dict[int, TaskInterface] = {}
        for index, (task_name, role) in enumerate(sorted(_TASK_ROLES.items())):
            handlers[index] = _DispatchTask(self, task_name)
            self.indexes[(_INSTANCE_NAMES[role], task_name)] = index
        self.manager.register_tasks(handlers)

    def close(self) -> None:
        for service in self.services.values():
            service.close()
        self.sdk.close()

    def start(
        self, instance_id: str, task_name: str, input_data: Mapping[str, Any]
    ) -> dict[str, Any]:
        index = self.indexes.get((instance_id, task_name))
        if index is None:
            raise ValueError(f"Ability 实例 {instance_id} 不支持 Task {task_name}")
        invocation_id = str(input_data.get("invocation_id", ""))
        if not invocation_id:
            raise ValueError("invocation_id 不能为空")
        self.counts[task_name] += 1
        started = self.manager.start_task(index, dict(input_data))
        task_id = str(started["task_id"])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            current = self.manager.get_task_info(task_id)
            if current and current["status"] == TaskStatus.COMPLETED:
                return {"task_id": task_id}
            if current and current["status"] == TaskStatus.FAILED:
                raise RuntimeError(
                    str(current.get("message") or "Ability Task 启动失败")
                )
            if current and current["status"] == TaskStatus.CANCELLED:
                raise RuntimeError("Ability Task 在登记 invocation 前被取消")
            time.sleep(0.002)
        raise TimeoutError(f"Ability Task {task_id} 未在期限内登记 invocation")

    def execute(self, task_name: str, input_data: Mapping[str, Any]) -> dict[str, Any]:
        role = _TASK_ROLES[task_name]
        invocation_id = str(input_data["invocation_id"])
        self.invocation_roles[invocation_id] = role
        injected = self._scenario_result(task_name, input_data)
        if injected is not None:
            self.synthetic[invocation_id] = injected
            return injected
        if task_name == "CloseUntilContact":
            # 完整 Mock 演示显式把目标放到夹爪接触范围内。Fake SDK 默认不伪造
            # 抓取成功，其他测试若未设置 fixture 仍会得到明确的未持物观测。
            for tool in input_data["tools"]:
                side = str(tool["tool_ref"]).rstrip("/").rsplit("/", 1)[-1]
                self.sdk.backend.configure_grasp_fixture(
                    side,
                    str(input_data["object_ref"]),
                    contact_opening_m=float(tool["target_position_m"]),
                    contact_force_n=float(tool["maximum_force_n"]),
                )
        return self.services[role].invoke(task_name, input_data)

    def get(
        self, instance_id: str, invocation_id: str, after_sequence: int
    ) -> dict[str, Any]:
        role = self._resolve_invocation(instance_id, invocation_id)
        record = self.synthetic.get(invocation_id)
        if record is not None:
            return {
                **record,
                "feedback": [
                    item
                    for item in record.get("feedback", ())
                    if int(item["sequence"]) > after_sequence
                ],
            }
        return self.services[role].invoke(
            "GetExecution",
            {"invocation_id": invocation_id, "after_feedback_sequence": after_sequence},
        )

    def stop(self, instance_id: str, invocation_id: str, reason: str) -> dict[str, Any]:
        role = self._resolve_invocation(instance_id, invocation_id)
        record = self.synthetic.get(invocation_id)
        if record is not None:
            stopped = {
                **record,
                "status": "stopped",
                "phase": "held",
                "result": {
                    "safe": True,
                    "physical_state": "base_stopped_and_braked",
                    "stop_evidence": {"holding": True, "stopped": True},
                },
                "error": None,
            }
            self.synthetic[invocation_id] = stopped
            return stopped
        return self.services[role].invoke(
            "StopExecution", {"invocation_id": invocation_id, "reason": reason}
        )

    def _resolve_invocation(self, instance_id: str, invocation_id: str) -> AbilityRole:
        role = self.invocation_roles.get(invocation_id)
        if role is None:
            raise KeyError(f"未知 invocation_id: {invocation_id}")
        if instance_id != _INSTANCE_NAMES[role]:
            raise ValueError(
                f"invocation {invocation_id} 不属于 Ability 实例 {instance_id}"
            )
        return role

    def _scenario_result(
        self, task_name: str, input_data: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """只注入验收需要的导航与槽位刷新故障；其余调用进入正式 Ability/SDK。"""

        invocation_id = str(input_data["invocation_id"])
        if task_name == "FollowRoute" and self.counts[task_name] == 1:
            observation = _observation(
                "route_blocked",
                "semantic://pallet-b/preplace",
                "route-block-1",
                {"blocked": True, "reason": "temporary_obstacle"},
            )
            return _execution(
                invocation_id,
                task_name,
                "running",
                feedback=[_feedback(1, "路线被临时障碍阻塞", observation)],
            )
        if task_name == "ObservePlacementTarget" and self.counts[task_name] == 1:
            target_ref = str(input_data["target_ref"])
            revision = "slot-occupied-1"
            observation = _observation(
                "placement.target_slot",
                target_ref,
                revision,
                {
                    "schema_version": 2,
                    "target_ref": target_ref,
                    "region_ref": target_ref,
                    "placement_pose": _pose(revision, 2.0, 1.0, 0.75),
                    "approach_vector": [0.0, 0.0, 1.0],
                    "extent_m": [0.56, 0.36, 0.02],
                    "free": False,
                    "reachable": True,
                    "occupants": ["object://mock/already-placed"],
                    "support_surface_ref": str(
                        input_data.get("target_ref") or "surface://pallet-b"
                    ),
                    "revision": revision,
                    "confidence": 0.98,
                    "evidence_refs": ["artifact://mock/slot-occupied"],
                },
            )
            return _execution(
                invocation_id,
                task_name,
                "succeeded",
                observations=[observation],
            )
        return None


def _execution(
    invocation_id: str,
    task_name: str,
    status: str,
    *,
    feedback: list[dict[str, Any]] | None = None,
    observations: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "invocation_id": invocation_id,
        "task_name": task_name,
        "operation": task_name,
        "command_id": f"mock-scenario:{invocation_id}",
        "resource": "scenario",
        "status": status,
        "progress": 0.5 if status == "running" else 1.0,
        "phase": status,
        "feedback_cursor": max(
            (item["sequence"] for item in feedback or ()), default=0
        ),
        "feedback": feedback or [],
        "observations": observations or [],
        "result": result or {},
        "error": error,
    }


def _feedback(
    sequence: int, message: str, observation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "status": "running",
        "phase": "blocked",
        "progress": 0.35,
        "message": message,
        "severity": "warning",
        "measurements": {},
        "observations": [observation],
        "evidence_refs": observation["evidence_refs"],
    }


def _observation(
    kind: str, subject_ref: str, revision: str, value: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": f"obs-{kind}-{revision}",
        "kind": kind,
        "schema_version": 2,
        "subject_ref": subject_ref,
        "source": "fake-scenario://depalletizing",
        "revision": revision,
        "frame_id": "map",
        "confidence": 0.98,
        "value": value,
        "artifact_refs": [],
        "evidence_refs": [f"artifact://mock/{kind}/{revision}"],
    }


def _pose(revision: str, x: float, y: float, z: float) -> dict[str, Any]:
    return {
        "frame_id": "map",
        "position_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "revision": revision,
    }


def _reply(
    message_id: Any, *, result: Any = None, error: Exception | None = None
) -> None:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": message_id}
    if error is None:
        message["result"] = result
    else:
        message["error"] = {"code": -32030, "message": str(error)}
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(gateway: MockAbilityGateway) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "ready"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params") or {}
            if method == "ability.start":
                result = gateway.start(
                    str(params["instance_id"]),
                    str(params["task_name"]),
                    params.get("input") or {},
                )
            elif method == "ability.get":
                result = gateway.get(
                    str(params["instance_id"]),
                    str(params["invocation_id"]),
                    int(params.get("after_sequence", 0)),
                )
            elif method == "ability.stop":
                result = gateway.stop(
                    str(params["instance_id"]),
                    str(params["invocation_id"]),
                    str(params.get("reason") or "pilot stop"),
                )
            elif method == "ability.stats":
                result = dict(gateway.counts)
            elif method == "ability.shutdown":
                _reply(request.get("id"), result={"ok": True})
                return
            else:
                raise ValueError(f"未知 RPC 方法: {method}")
            _reply(request.get("id"), result=result)
        except Exception as exc:  # noqa: BLE001 - RPC 边界必须返回结构化错误
            _reply(request.get("id") if "request" in locals() else None, error=exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 Pro Ability 完整 Mock 验收进程")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--store-dir", required=True)
    args = parser.parse_args()
    gateway = MockAbilityGateway(
        Path(args.profile), Path(args.models), Path(args.store_dir)
    )
    try:
        serve(gateway)
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
