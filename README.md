# R1 Pro Abilities

[English](README.md) | [简体中文](README.zh-CN.md)

🤖 Semantic abilities for R1 Pro: navigation, manipulation, end-effector control, state, sensing, perception, and grasp planning. AbilityFramework hosts the processes; Robot SDK provides hardware/simulation access. Task-level Skills live in the separate robot-skill repository.

## Structure

- `r1pro_abilities/` — shared Python implementation.
- `abilities/` — seven Ability entry points and manifests.
- `configs/` — Robot configuration examples.
- `tests/` — behavior and contract tests.

## 🛠 Build

Python **3.11+** is required; the current quick-start Robot Bundle uses **3.13**. Build ability-py-sdk and robot-sdk first. In the quick-start layout:

```bash
uv venv --python 3.13
uv pip install ../../ability-framework/ability-py-sdk/dist/*.whl \
  ../../semantic-robotsdk/robot-sdk/dist/*.whl
uv pip install -e .
make build ROBOT_SDK_PATH=../../semantic-robotsdk/robot-sdk
```

Use a consistent manifest version and keep only its intended Wheels in the input `dist/` directories. `make build` runs tests and produces `dist/semantic_r1pro_abilities-*.whl`.

## Use the outputs

The shared Wheel is not a complete Ability deployment. The framework's `scripts/refresh_v050_mujoco.py` packages the seven Ability projects, collects SDK Wheels, and assembles them into a Robot Bundle through semantic-deployment.

Set `SEMANTIC_ROBOT_CONFIG` for each Robot instance. Use Fake/simulation configurations for development; connecting real hardware requires the appropriate driver, calibration, and safety checks.

## Troubleshooting

- Import failures usually mean missing local SDK Wheels or the wrong Python environment.
- The Makefile's default SDK path differs from quick-start's nested layout; pass `ROBOT_SDK_PATH` as shown.
- A successful Wheel build does not start abilities or publish Skills.
- Changes to manifests, SDK contracts, and bundle versions must be tested together.

[Detailed technical reference](README.reference.md) · [Ability projects](abilities/) · [Robot configuration](configs/)

## License

Copyright 2026 InsightOS. First-party code: [Apache-2.0](LICENSE). See [NOTICE](NOTICE) and [license scope](LICENSE_SCOPE.md) for third-party components and assets.

## Reproducible platform builds

See [glibc, musl and macOS build instructions](README.build.md) for pinned source revisions, exact scripts, tool requirements, local commands, CI reproduction and platform support boundaries.
