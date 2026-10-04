"""AbilityFramework Mock 进程的实例路由和正式 Task 回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mock_gateway import MockAbilityGateway


ROOT = Path(__file__).resolve().parents[1]


class MockAbilityGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.gateway = MockAbilityGateway(
            ROOT / "configs/robot-deployment.fake.yaml",
            ROOT / "configs/models.example.json",
            Path(self.temp.name),
        )

    def tearDown(self) -> None:
        self.gateway.close()
        self.temp.cleanup()

    def test_exact_instance_runs_formal_task_and_rejects_cross_binding(self) -> None:
        invocation_id = "plan-route-gateway-1"
        started = self.gateway.start(
            "fake-r1-navigation",
            "PlanRoute",
            {
                "invocation_id": invocation_id,
                "robot_id": "r1pro-fake-001",
                "target": {
                    "target_ref": "semantic://pallet-b/preplace",
                    "pose": {
                        "frame_id": "world",
                        "position_m": [2.0, 1.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "revision": "target-r1",
                    },
                    "constraints": {},
                },
                "navigation_purpose": "transit",
                "maximum_speed_mps": 0.5,
                "minimum_clearance_m": 0.25,
            },
        )
        self.assertTrue(started["task_id"])

        result = self.gateway.get("fake-r1-navigation", invocation_id, 0)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["result"]["route_ref"])
        self.assertEqual(self.gateway.counts["PlanRoute"], 1)

        with self.assertRaises(ValueError):
            self.gateway.get("fake-r1-object-perception", invocation_id, 0)


if __name__ == "__main__":
    unittest.main()
