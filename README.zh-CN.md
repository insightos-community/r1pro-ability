# R1 Pro Abilities

[English](README.md) | [简体中文](README.zh-CN.md)

🤖 R1 Pro 的语义能力：导航、机械臂运动、末端控制、状态、传感、感知与抓取规划。进程由 AbilityFramework 托管，硬件 / 仿真访问由 Robot SDK 提供；任务级 Skill 位于独立的 robot-skill 仓库。

## 工程结构

- `r1pro_abilities/`：共享 Python 实现。
- `abilities/`：七类 Ability 入口与清单。
- `configs/`：Robot 配置示例。
- `tests/`：行为与契约测试。

## 🛠 构建

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
