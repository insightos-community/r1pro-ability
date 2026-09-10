# CI and releases

PRs and main pushes validate and package `r1pro-ability` on GitHub-hosted Ubuntu 24.04. Version tag pushes publish a release. For existing tags, run **CI and Release** on **main** with the exact tag as input. Published releases are never overwritten and tags are never moved.

Release assets include component payloads, Python wheels where applicable, `release.json` with source/tag and build recipe SHA, and `SHA256SUMS`. Consumers must verify checksums and compare the source commit with quick-start's `repo-versions.json`. Python package versions follow tagged pyproject metadata. Build commands and payload selection are in `.github/scripts/`.

CI covers component tests or package integrity/smoke checks, not GPU, real hardware or full-stack acceptance. Data packages retain the tagged asset notices and third-party licenses. `ability-runtime` contains the pinned third-party wheel cache; first-party binaries and wheels come from their own releases.

中文：推送版本 Tag 自动发布，历史 Tag 可从 main 手动补发。installer 按版本清单下载并校验产物；能力和技能仓库同时提供可安装 ZIP。数据仓库检查 LFS 实体与归档完整性。
