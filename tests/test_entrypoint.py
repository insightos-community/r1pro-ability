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

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from r1pro_abilities.entrypoint import RuntimeConfig, run_ability
from r1pro_abilities.service import AbilityRole, R1ProAbilityService
from semantic_robot_sdk_core import SensorFrame, load_robot_deployment
from semantic_robot_sdk_r1pro import R1ProSDK


class _FakeTaskManager:
    def __init__(self) -> None:
        self.tasks = {}

    def register_tasks(self, tasks) -> None:
        self.tasks = dict(tasks)


class _FakeServer:
    def __init__(self) -> None:
        self.shutdown_called = False

    def serve_forever(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_called = True


class AbilityEntrypointTest(unittest.TestCase):
    def test_runtime_config_uses_one_robot_config_and_per_instance_store(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = root / "configs" / "robot-deployment.fake.yaml"
        ability_root = root / "abilities" / "r1pro-navigation"
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "ABILITY_ROOT": str(ability_root),
                "SEMANTIC_ROBOT_CONFIG": str(profile),
                "SEMANTIC_ABILITY_EXECUTION_ROOT": temporary,
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(sys, "argv", ["main.py", "instance-7", "{}"]),
            ):
                config = RuntimeConfig.load("R1ProNavigation.V2")

            self.assertEqual(config.robot_deployment_path, str(profile))
            self.assertEqual(
                config.execution_store_path,
                str(Path(temporary) / "instance-7" / "r1pro-navigation.sqlite"),
            )
            self.assertTrue(Path(config.execution_store_path).parent.is_dir())

    def test_sensor_capture_entrypoint_returns_refs_relative_to_shared_robot_root(self) -> None:
        """真实环境配置入口不可在 Pilot 不知情时额外嵌套 Ability UUID。"""

        root = Path(__file__).resolve().parents[1]
        profile = root / "configs" / "robot-deployment.fake.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            shared_root = Path(temporary) / "robot-1" / "artifact-exchange"
            execution_root = Path(temporary) / "executions"
            environment = {
                "ABILITY_ROOT": str(root / "abilities" / "r1pro-sensor-capture"),
                "SEMANTIC_ROBOT_CONFIG": str(profile),
                "SEMANTIC_ABILITY_ARTIFACT_ROOT": str(shared_root),
                "SEMANTIC_ABILITY_EXECUTION_ROOT": str(execution_root),
            }
            configs = []
            for instance_id in ("sensor-instance-1", "sensor-instance-2"):
                with (
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.object(sys, "argv", ["main.py", instance_id, "{}"]),
                ):
                    configs.append(RuntimeConfig.load("R1ProSensorCapture.V2"))
            # Windows temp paths may arrive in 8.3 form while RuntimeConfig
            # expands them. Compare directories, still rejecting an extra UUID.
            self.assertEqual(Path(configs[0].artifact_exchange_root).resolve(), shared_root.resolve())
            self.assertEqual(Path(configs[1].artifact_exchange_root).resolve(), shared_root.resolve())
            self.assertNotEqual(configs[0].execution_store_path, configs[1].execution_store_path)
            self.assertIn("sensor-instance-1", configs[0].execution_store_path)
            self.assertIn("sensor-instance-2", configs[1].execution_store_path)

            sdk = R1ProSDK.from_deployment(load_robot_deployment(configs[0].robot_deployment_path))
            service = R1ProAbilityService(
                AbilityRole.SENSOR_CAPTURE, sdk,
                execution_store_path=configs[0].execution_store_path,
                artifact_exchange_root=configs[0].artifact_exchange_root,
            )
            try:
                frame_bytes = b"captured-frame-from-fake-sensor"
                sdk.backend.publish_sensor_frame(SensorFrame(
                    sensor_id="rgb", sequence=7,
                    generation=sdk.state.snapshot().generation,
                    frame_id="camera", encoding="jpeg", payload=frame_bytes,
                    width=640, height=480,
                ))
                paths = []
                for invocation in ("capture-stage-1", "capture-stage-2"):
                    result = service.invoke("CaptureRGBD", {
                        "robot_id": sdk.robot_id, "sensor_ids": ["rgb"],
                        "invocation_id": invocation,
                    })
                    self.assertEqual(result["status"], "succeeded")
                    candidate = result["result"]["artifact_candidates"][0]
                    relative = Path(candidate["exchange_path"])
                    self.assertFalse(relative.is_absolute())
                    self.assertEqual(relative.parts[0], "captures")
                    captured = shared_root / relative  # Same resolution as Pilot.
                    self.assertEqual(captured.read_bytes(), frame_bytes)
                    self.assertEqual(candidate["media_type"], "image/jpeg")
                    paths.append(captured)
                self.assertNotEqual(paths[0].parent, paths[1].parent)
                self.assertFalse((shared_root / "sensor-instance-1").exists())
            finally:
                service.close()
                sdk.close()

    def test_unconfigured_exchange_root_remains_unavailable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with mock.patch.dict(os.environ, {
            "SEMANTIC_ROBOT_CONFIG": str(root / "configs" / "robot-deployment.fake.yaml"),
            "EXECUTION_STORE_PATH": ":memory:",
        }, clear=True):
            config = RuntimeConfig.load("R1ProSensorCapture.V2")
        self.assertIsNone(config.artifact_exchange_root)

    def test_ability_lifecycle_owns_sdk_and_task_server(self) -> None:
        task_manager = _FakeTaskManager()
        captured = {}

        ability_py = types.ModuleType("ability_py")
        task_server = types.ModuleType("ability_py.task_server")
        task_server.app = object()

        class AbilityInterface:
            pass

        class TaskInterface:
            pass

        class AbilityService:
            def run(self, lifecycle) -> None:
                lifecycle.on_connect()
                captured["port"] = lifecycle.get_ability_port()
                captured["sdk"] = lifecycle.sdk
                captured["service"] = lifecycle.service
                captured["server"] = lifecycle.task_server._server
                captured["invalid_task"] = task_manager.tasks[0].execute({})
                lifecycle.on_disconnect()
                captured["lifecycle_sdk_after_disconnect"] = lifecycle.sdk

        ability_py.AbilityInterface = AbilityInterface
        ability_py.TaskInterface = TaskInterface
        ability_py.AbilityService = AbilityService
        ability_py.task_manager = task_manager
        ability_py.get_free_port = lambda: 18081

        werkzeug = types.ModuleType("werkzeug")
        serving = types.ModuleType("werkzeug.serving")
        serving.make_server = lambda *_args: _FakeServer()

        profile = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "robot-deployment.fake.yaml"
        )
        modules = {
            "ability_py": ability_py,
            "ability_py.task_server": task_server,
            "werkzeug": werkzeug,
            "werkzeug.serving": serving,
        }
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                RuntimeConfig,
                "load",
                return_value=RuntimeConfig(str(profile), None, ":memory:"),
            ),
        ):
            run_ability(AbilityRole.NAVIGATION, "R1ProNavigation.V2")

        self.assertEqual(captured["port"], 18081)
        self.assertIsNotNone(captured["sdk"])
        self.assertIsNotNone(captured["service"])
        self.assertTrue(captured["server"].shutdown_called)
        self.assertIsNone(captured["lifecycle_sdk_after_disconnect"])
        self.assertEqual(len(task_manager.tasks), 5)
        self.assertEqual(
            captured["invalid_task"]["request_rejected"]["code"],
            "INVALID_TASK_INPUT",
        )


if __name__ == "__main__":
    unittest.main()
