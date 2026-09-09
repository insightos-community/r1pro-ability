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

"""使用现有 ability_py.task_manager 注册 Task 的薄启动层。"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from semantic_robot_sdk_core import load_robot_deployment
from semantic_robot_sdk_r1pro import R1ProSDK

from .models import ModelProfileRegistry
from .service import TASKS, AbilityRole, R1ProAbilityService


@dataclass(frozen=True)
class RuntimeConfig:
    robot_deployment_path: str
    model_registry_path: str | None
    execution_store_path: str
    artifact_exchange_root: str | None = None

    @classmethod
    def load(cls, ability_name: str) -> "RuntimeConfig":
        # 一台 Robot 的部署配置由 Pilot/AbilityFramework 环境统一提供；CR 只保存该
        # Ability 自己的默认值。这样新增 Robot 时不需要逐个修改七份 CR。
        cr_path = _find_cr(ability_name)
        config: dict[str, Any] = {}
        if cr_path is not None:
            data = yaml.safe_load(cr_path.read_text(encoding="utf-8")) or {}
            config = (data.get("spec") or {}).get("config") or {}

        robot_deployment = os.environ.get("SEMANTIC_ROBOT_CONFIG") or config.get(
            "robotDeploymentPath"
        )
        if not robot_deployment:
            raise RuntimeError(
                f"没有找到 {ability_name} 的 CR，也未设置 SEMANTIC_ROBOT_CONFIG"
            )
        model_registry = os.environ.get("MODEL_REGISTRY_PATH") or config.get(
            "modelRegistryPath"
        )
        return cls(
            _resolve_path(str(robot_deployment), cr_path),
            _resolve_path(str(model_registry), cr_path) if model_registry else None,
            _execution_store_path(
                str(config.get("executionStorePath", ":memory:")), cr_path
            ),
            _artifact_exchange_root(),
        )


def _resolve_path(value: str, cr_path: Path | None) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    package_root = cr_path.parent.parent if cr_path is not None else Path.cwd()
    return str((package_root / path).resolve())


def _execution_store_path(configured: str, cr_path: Path | None) -> str:
    explicit = os.environ.get("EXECUTION_STORE_PATH")
    if explicit:
        return str(Path(explicit).expanduser()) if explicit != ":memory:" else explicit
    if configured == ":memory:":
        return configured

    root = os.environ.get("SEMANTIC_ABILITY_EXECUTION_ROOT")
    if root:
        # AbilityFramework 把实例 UUID 放在第一个参数。不同实例必须拥有不同数据库，
        # 否则一个实例停止或恢复时会读到另一个实例的 invocation。
        instance_id = sys.argv[1] if len(sys.argv) > 1 else "standalone"
        path = Path(root).expanduser() / instance_id / Path(configured).name
    else:
        path = Path(_resolve_path(configured, cr_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _artifact_exchange_root() -> str | None:
    """返回 Ability 与 Pilot 共享的二进制交换目录。

    AbilityFramework 启动的七个 Ability 是独立进程，二进制帧不能经 Task
    JSON 返回。实例启动器已按 Robot 隔离根目录，生产者返回的 exchange_path
    必须相对于这个相同根目录，不能再隐式追加 Ability UUID。SensorCapture
    在根目录内按 invocation 摘要分目录；Execution 数据库仍按 Ability 实例隔离。
    """

    root = os.environ.get("SEMANTIC_ABILITY_ARTIFACT_ROOT")
    if not root:
        return None
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _find_cr(ability_name: str) -> Path | None:
    explicit = os.environ.get("ABILITY_CR_PATH")
    if explicit:
        return Path(explicit)
    roots = [
        Path(os.environ[key])
        for key in ("ABILITY_ROOT", "ABILITY_HOME")
        if os.environ.get(key)
    ]
    roots.extend([Path.cwd(), Path(__file__).resolve().parent.parent])
    for root in roots:
        for path in root.glob("**/crs/*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if ((data.get("spec") or {}).get("abilityName")) == ability_name:
                return path
    return None


def run_ability(role: AbilityRole, ability_name: str) -> None:
    """注册 Task 并交还给官方 AbilityService；不接管进程或任务调度。"""

    import ability_py
    from ability_py import TaskInterface, task_manager
    from ability_py.task_server import app
    from werkzeug.serving import make_server

    class ServerThread(threading.Thread):
        """沿用 AbilityFramework 的任务服务，不创建第二套运行时。"""

        def __init__(self, port: int) -> None:
            super().__init__(daemon=True)
            self._server = make_server("0.0.0.0", port, app)

        def run(self) -> None:
            self._server.serve_forever()

        def shutdown(self) -> None:
            self._server.shutdown()

    class Lifecycle(ability_py.AbilityInterface):
        def __init__(self) -> None:
            self.ability_port = 0
            self.task_server: ServerThread | None = None
            self.sdk: R1ProSDK | None = None
            self.service: R1ProAbilityService | None = None

        def on_start(self) -> None:
            return None

        def on_connect(self) -> None:
            config = RuntimeConfig.load(ability_name)
            deployment = load_robot_deployment(config.robot_deployment_path)
            self.sdk = R1ProSDK.from_deployment(deployment)
            models = (
                ModelProfileRegistry.load(config.model_registry_path)
                if config.model_registry_path
                else None
            )
            self.service = R1ProAbilityService(
                role,
                self.sdk,
                models,
                config.execution_store_path,
                config.artifact_exchange_root,
            )
            if self.ability_port == 0:
                self.ability_port = ability_py.get_free_port()
                self.task_server = ServerThread(self.ability_port)
                self.task_server.start()

        def on_disconnect(self) -> None:
            if self.task_server is not None:
                self.task_server.shutdown()
            self.task_server = None
            self.ability_port = 0
            if self.service is not None:
                self.service.close()
            if self.sdk is not None:
                self.sdk.close()
            self.sdk = None
            self.service = None

        def on_terminate(self) -> None:
            self.on_disconnect()

        def get_ability_port(self) -> int:
            return self.ability_port

    class DispatchTask(TaskInterface):
        def __init__(self, lifecycle: Lifecycle, task_name: str) -> None:
            self._lifecycle = lifecycle
            self._task_name = task_name

        def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
            if self._lifecycle.service is None:
                raise RuntimeError("Ability 尚未连接 Robot SDK")
            try:
                return self._lifecycle.service.invoke(self._task_name, input_data)
            except ValidationError as exc:
                # AbilityFramework 先登记异步 Task，随后才在线程中调用这里。因此
                # Pydantic 参数错误不会表现为 HTTP 4xx。用结构化回执明确告诉 Pilot
                # “校验阶段已经拒绝、尚未启动 SDK 命令”，Pilot 才能安全释放 Robot；
                # 其他运行期异常仍由 Framework 保留为失败，不能误报成启动前拒绝。
                return {
                    "request_rejected": {
                        "code": "INVALID_TASK_INPUT",
                        "message": str(exc),
                    }
                }

    lifecycle = Lifecycle()
    task_names = (*TASKS[role], "GetExecution", "StopExecution")
    task_manager.register_tasks(
        {
            task_type: DispatchTask(lifecycle, task_name)
            for task_type, task_name in enumerate(task_names)
        }
    )
    ability_py.AbilityService().run(lifecycle)
