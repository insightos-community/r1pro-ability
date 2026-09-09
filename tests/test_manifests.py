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

import importlib
import os
import unittest
from pathlib import Path

import yaml

from r1pro_abilities.service import TASKS, AbilityRole


ROOT = Path(__file__).resolve().parents[1]
ABILITY_DIRS = {
    AbilityRole.NAVIGATION: "r1pro-navigation",
    AbilityRole.MANIPULATOR_MOTION: "r1pro-manipulator-motion",
    AbilityRole.END_EFFECTOR: "r1pro-end-effector",
    AbilityRole.ROBOT_STATE: "r1pro-robot-state",
    AbilityRole.SENSOR_CAPTURE: "r1pro-sensor-capture",
    AbilityRole.OBJECT_PERCEPTION: "r1pro-object-perception",
    AbilityRole.GRASP_PLANNING: "r1pro-grasp-planning",
}


class AbilityManifestTest(unittest.TestCase):
    def test_seven_manifests_match_registered_tasks(self) -> None:
        self.assertEqual(len(ABILITY_DIRS), 7)
        for role, directory in ABILITY_DIRS.items():
            manifest_path = ROOT / "abilities" / directory / "ability.manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            task_names = tuple(task["taskName"] for task in manifest["tasks"])
            task_types = tuple(task["taskType"] for task in manifest["tasks"])
            self.assertEqual(task_types, tuple(range(len(task_types))))
            self.assertEqual(task_names[: len(TASKS[role])], TASKS[role])
            self.assertEqual(task_names[-2:], ("GetExecution", "StopExecution"))

            # 语义 Action 由各 Ability 自己的 Manifest 声明。Pilot 只读取
            # 这里的元数据做精确实例路由，不再维护机器人型号专用映射表。
            business_tasks = manifest["tasks"][: len(TASKS[role])]
            action_types = {task.get("actionType") for task in business_tasks}
            self.assertNotIn(None, action_types)
            self.assertEqual(len(action_types), len(business_tasks))
            for task in business_tasks:
                self.assertEqual(task.get("abilityRole"), role.value)
                self.assertEqual(task.get("schemaVersion"), 2)
                self.assertIsInstance(task.get("physical"), bool)
            for task in manifest["tasks"][-2:]:
                self.assertNotIn("actionType", task)

            self.assertTrue(all(task.get("inputModel") for task in manifest["tasks"]))
            for task in manifest["tasks"]:
                module_name, _, model_name = task["inputModel"].partition(":")
                self.assertTrue(module_name and model_name)
                model = getattr(importlib.import_module(module_name), model_name)

                # Robot/Pilot 在调用时补齐 invocation_id 与 robot_id；设备页只展示
                # Robot Skill 或调试人员需要填写的业务参数。Manifest 中的字段文档
                # 必须与 Pydantic 模型同步，避免界面展示一套、Ability 校验另一套。
                if task in business_tasks:
                    documented = {item["name"]: item for item in task["inputFields"]}
                    business_fields = {
                        name: field
                        for name, field in model.model_fields.items()
                        if name not in {"invocation_id", "robot_id"}
                    }
                    self.assertEqual(set(documented), set(business_fields))
                    for name, field in business_fields.items():
                        self.assertEqual(documented[name]["required"], field.is_required())
                        self.assertTrue(documented[name]["description"].strip())

            cr_path = ROOT / "abilities" / directory / "crs" / f"{directory}.yaml"
            cr = yaml.safe_load(cr_path.read_text(encoding="utf-8"))
            config = cr["spec"]["config"]
            self.assertEqual(
                config["robotDeploymentPath"],
                "configs/robot-deployment.fake.yaml",
            )
            self.assertNotIn("robotProfilePath", config)

            launcher = ROOT / "abilities" / directory / "bin" / "ability"
            requirements = ROOT / "abilities" / directory / "requirements.txt"
            self.assertTrue(launcher.is_file())
            self.assertTrue(os.access(launcher, os.X_OK))
            self.assertTrue(requirements.is_file())
            self.assertIn("ability-py==0.4.0", requirements.read_text(encoding="utf-8"))

    def test_business_inputs_do_not_expose_deployment_fields(self) -> None:
        forbidden = {
            "robotUri",
            "topic",
            "endpoint",
            "firmware",
            "model_path",
            "weights_path",
        }
        for model in __import__(
            "r1pro_abilities.task_models", fromlist=["TASK_INPUTS"]
        ).TASK_INPUTS.values():
            self.assertFalse(set(model.model_fields) & forbidden)


if __name__ == "__main__":
    unittest.main()
