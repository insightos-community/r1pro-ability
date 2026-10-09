# Semantic R1 Pro Abilities

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库保留 AbilityFramework 的 Manifest、CR、Task 注册和生命周期形式。Ability
通过进程内 `semantic_robot_sdk_r1pro` 使用 Robot，不读取 ROS Topic、设备 IP、固件差异或仿真私有接口，
也不依赖旧 `skill_library`。进程由 AbilityFramework 托管，硬件 / 仿真访问由 Robot SDK 提供；任务级 Skill 位于独立的 robot-skill 仓库。

v0.5 按共享资源、依赖、停止方式和升级周期划分为七个逻辑接口：

- Navigation：路线计算、路径跟随和到达复核。
- ManipulatorMotion：末端短运动、轨迹跟随和携物抬升。
- EndEffector：受控闭合、释放和保持。
- RobotState：Robot 状态与持物状态。
- SensorCapture：RGBD 获取与 Artifact 引用。
- ObjectPerception：目标、预抓取、抓取和放置结果的感知与复核。
- GraspPlanning：抓取候选生成和排序。

周转箱链使用 Action schema v2：运动输入为左右末端的 `targets[]`，末端执行器
输入为 `tools[]`。MuJoCo Provider 从 Runtime 的真实对象、工具接触和 RGB-D
状态形成 Observation；不会使用 Fake Provider、隐藏 attach 或直接写物体 Pose。

`PolicyControl` 属于 v0.8 的模型闭环能力，不进入 v0.5 的包、CR 和运行进程。

每个业务输入由 Pydantic 严格模型检查，不能携带连接和部署字段。每次调用保存
invocation、底层 command、按序 Feedback、Observation、结果与错误；`GetExecution`
支持从指定反馈序号继续读取。`StopExecution` 会把停止传到 Robot SDK，并且只有取得
设备停止和 hold 证据后才报告 `stopped`。

每个业务 Task 还在同一份 `ability.manifest.yaml` 中声明 `inputModel`、
`inputFields` 和 `returns`。`inputFields` 只描述 Robot Skill 或调试人员需要提供的
语义参数，`robot_id` 与 `invocation_id` 由 Pilot 补齐。设备页和右侧 Inspector
直接展示这些字段的名称、类型、必填性和说明；新增或修改 Pydantic 输入字段时必须
同步更新 Manifest，`tests/test_manifests.py` 会阻止二者发生漂移。

AbilityFramework 二进制和 `ability_py` wheel 由部署环境提供，运行数据库和日志不提交。
同一台 Robot 的 Backend Endpoint、固件 Profile、Provider 和安全参数只保存在一份
`RobotDeployment` YAML 中。各 CR 的 `spec.config` 只引用同一个
`robotDeploymentPath`，并保存本 Ability 自己的 `executionStorePath` 和可选
`modelRegistryPath`；部署新 Robot 时不再逐个复制底层连接参数。

## 工程结构

- `r1pro_abilities/`：共享 Python 实现。
- `abilities/`：七类 Ability 入口与清单。
- `configs/`：Robot 配置示例。
- `tests/`：行为与契约测试。

## 构建、打包和部署

七类 Ability 是七个独立 AbilityFramework 包，共享一份业务 Wheel：

```text
dist/semantic_r1pro_abilities-*.whl
abilities/
├── r1pro-navigation/
├── r1pro-manipulator-motion/
├── r1pro-end-effector/
├── r1pro-robot-state/
├── r1pro-sensor-capture/
├── r1pro-object-perception/
└── r1pro-grasp-planning/
```

AbilityFramework 的 Python 环境通过 Wheel 安装公共实现和 Robot SDK，不从源码目录加载：

```bash
make build
python -m pip install \
  dist/semantic_r1pro_abilities-*.whl \
  /path/to/semantic_robot_sdk_core-*.whl \
  /path/to/semantic_robot_sdk_r1pro-*.whl
```

每个 `abilities/<name>` 目录使用提供的 `ability-scaffold pack` 在 staging 目录打包。
源码目录不保存 Framework 运行数据库、实例日志或打包工具生成的文件。

机器人类型包共享上述 Wheel 和七个 Ability Zip，但每台 Robot 启动独立
AbilityFramework。实例启动器为该 Robot 生成一份 `RobotDeployment`、七份 CR
运行配置和独立 Execution 数据目录，再通过可配置的 AbilityFramework Endpoint 上传、
启动和查询实例。旧 `ability_tool` 只适合默认 8080 的单实例开发环境，不用于同机多
Robot 编排。

Ability 进程读取 `SEMANTIC_ROBOT_CONFIG`，并正常 Import 已安装的
`semantic_robot_sdk_r1pro`。同型号 Robot 可以共享相同制品；不同 Robot 的
`robot.id`、AbilityFramework Endpoint、Execution Store 和 Robot SDK 连接配置不能
共用。

SensorCapture 还读取由实例启动器设置的 `SEMANTIC_ABILITY_ARTIFACT_ROOT`。真实
RGB/Depth 帧先原子写入该目录，Task JSON 只返回相对 `exchange_path`；Pilot 校验并
上传后才生成 Server ArtifactRef。未配置交换目录时 CaptureRGBD 明确失败，不返回伪
`artifact://`。

该交换根目录已由启动器按 Robot 隔离；Ability 不再隐式追加自身实例 UUID。
`exchange_path` 始终可由 Pilot 直接相对于同一个根目录解引用，采图文件仍按
invocation 摘要隔离。Ability Execution 数据库的实例隔离保持不变。

自动测试仍使用 Fake Backend 验证执行、停止和 schema，但正式 MuJoCo 类型包使用
`mujoco_ground_truth` 和 `mujoco-tote-grasp` Provider。Isaac Backend 在 v0.5
中明确返回未实现。

## 源码开发环境

需要 Python **3.11+**；当前 quick-start Robot Bundle 使用 **3.13**。先构建 ability-py-sdk 和 robot-sdk，再按 quick-start 布局执行：

```bash
uv venv --python 3.13
uv pip install ../../ability-framework/ability-py-sdk/dist/*.whl \
  ../../semantic-robotsdk/robot-sdk/dist/*.whl
uv pip install -e .
make build ROBOT_SDK_PATH=../../semantic-robotsdk/robot-sdk
```

各组件应使用同一清单锁定的版本，输入 `dist/` 中只保留对应 Wheel。`make build` 会运行测试，生成 `dist/semantic_r1pro_abilities-*.whl`。

## 产物使用

共享 Wheel 不是完整的 Ability 部署包。Framework 的 `scripts/refresh_v050_mujoco.py` 会打包七个 Ability 工程、收集 SDK Wheel，并通过 semantic-deployment 组装 Robot Bundle。

为每个 Robot 实例设置 `SEMANTIC_ROBOT_CONFIG`。开发优先使用 Fake / 仿真配置；真机需要对应驱动、标定与安全检查。

## 常见问题

- 导入失败通常是缺少本地 SDK Wheel，或使用了错误的 Python 环境。
- Makefile 默认 SDK 路径与 quick-start 嵌套布局不同，请显式传入示例中的 `ROBOT_SDK_PATH`。
- Wheel 构建成功不代表 Ability 已启动，也不代表 Skill 已发布。
- 清单、SDK 契约与 Bundle 版本变更需要联合验证。

[详细技术参考](README.reference.md) · [Ability 工程](abilities/) · [Robot 配置](configs/)

## 许可证

Copyright 2026 InsightOS。自有代码采用 [Apache-2.0](LICENSE)；第三方组件与资产请查看 [NOTICE](NOTICE) 和[许可范围](LICENSE_SCOPE.md)。

## 三个平台的构建复现

参见 [glibc、musl 与 macOS 构建说明](README.build.md)：包含已锁定的源码版本、实际脚本入口、工具要求、本地与 CI 指令、产物位置和平台验证范围。
