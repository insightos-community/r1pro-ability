# r1pro-ability: reproducible platform builds

## Versions, tools and build layout

The glibc installer pins component tag **v0.4.0-insightos.2026.2** at `76e670f0cb8874b12fadbeb9acc9a88da7fb9b9c`.
This guide pins the current build-script snapshot at `ac6dc056448ae19f0ed6e588b7f58b443a1d1a9c`.
To reconstruct another published release, read its `release.json` and select
both `source_commit` and `build_recipe_commit`; a source tag alone may predate
the CI scripts. This recipe reproduces the build steps, not historical archive bytes.

Prerequisites: uv 0.12.12, Git, Make, an authenticated GitHub CLI (`gh auth login` or `GH_TOKEN`) and Python 3; component scripts select Python 3.13. Use a fresh virtual environment for each platform.

The release scripts expect **two sibling checkouts**, `automation/` for build
scripts and `source/` for the component. Run these commands from a fresh working
directory (the scripts themselves are not standalone copies):

```bash
REPRO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/r1pro-ability-repro.XXXXXXXX")"
git clone --no-checkout https://github.com/insightos-community/r1pro-ability.git "$REPRO_ROOT/automation"
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPRO_ROOT/automation" checkout --detach ac6dc056448ae19f0ed6e588b7f58b443a1d1a9c
git clone --no-checkout https://github.com/insightos-community/r1pro-ability.git "$REPRO_ROOT/source"
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPRO_ROOT/source" checkout --detach v0.4.0-insightos.2026.2
cd "$REPRO_ROOT/source"
test "$(git rev-parse HEAD)" = 76e670f0cb8874b12fadbeb9acc9a88da7fb9b9c
export TARGET_TAG=v0.4.0-insightos.2026.2
export COMPONENT=r1pro-ability
export GITHUB_SHA=ac6dc056448ae19f0ed6e588b7f58b443a1d1a9c
```

## Linux glibc / standard component Release

The executable build entry is [`.github/scripts/build.sh`](.github/scripts/build.sh);
archive validation is [`.github/scripts/package.py`](.github/scripts/package.py).
The dependency download script uses `gh release download` to fetch and verify
pinned SDK/scaffold wheels. Confirm authentication before starting the local build.
From `source/` in the layout above:

```bash
gh auth status
bash ../automation/.github/scripts/build.sh
python3 ../automation/.github/scripts/package.py 
(cd .output/release && sha256sum -c SHA256SUMS)
```

Artifacts: `source/.output/release/` (archives/wheels, `release.json`, checksum
inventory and license notices). `release.json` records source and recipe revisions.
The local commands do not publish or overwrite a GitHub Release.

## Linux musl

This component produces pure Python wheels, scripts, static Web/docs or assets
that are reused by the musl installer. Reproduce the component on the standard
build host above; a second musl compilation of those same files is unnecessary.
Native transitive dependencies must still be obtained from the musl lock.

For the complete musl build and offline checks, use the [quick-start musl commands](https://github.com/insightos-community/quick-start/blob/main/README.build.md#linux-musl-x86_64).

## macOS / macosx

The component wheel is `py3-none-any` and is reused in the macOS installer.
On an Apple Silicon build host with uv 0.12.12, build the pure package from a
fresh checkout at the source revision above:

```bash
uv build
# Output: dist/*.whl. Install pinned sibling SDK wheels before testing.
```

Install the pinned sibling SDK wheels and native macOS dependency set before
running application tests; do not resolve unreleased `semantic-*` packages from
PyPI. The full installer recipe downloads those component wheels explicitly.

The complete macOS installer targets Apple Silicon/macOS 15.5+; see the [locked assembly instructions](https://github.com/insightos-community/quick-start/blob/main/README.build.md#macos-apple-silicon).

## GitHub workflow reproduction

The repository’s [CI workflow](.github/workflows/ci.yml) implements the two-checkout
layout. To build a source tag without publishing, create a reproduction branch at the
pinned automation commit. GitHub dispatch expects a branch/tag ref; both tag refs
and default-branch dispatches can enter this workflow’s publishing job. The following
commands require repository write access and use a non-default branch:

```bash
gh auth setup-git
REPRO_BRANCH=reproduce/platform-builds
git -C "$REPRO_ROOT/automation" push origin ac6dc056448ae19f0ed6e588b7f58b443a1d1a9c:refs/heads/$REPRO_BRANCH
gh workflow run ci.yml --repo insightos-community/r1pro-ability --ref "$REPRO_BRANCH" -f tag=v0.4.0-insightos.2026.2
gh run list --repo insightos-community/r1pro-ability --workflow ci.yml --limit 5
# Set REPRO_RUN_ID to the selected run ID.
gh run watch "$REPRO_RUN_ID" --repo insightos-community/r1pro-ability --exit-status
gh run download "$REPRO_RUN_ID" --repo insightos-community/r1pro-ability --name release-assets --dir downloaded-release
```

## Reproduction evidence

Build in a fresh checkout and a separate output directory for each ABI. Preserve
source commits, compiler/tool versions, dependency locks, package inventories and
test logs. Fixed source revisions and a container digest reproduce the recipe;
unlocked OS packages, runner images, timestamps and build tools can still change
archive bytes. Compare a downloaded release against its published `SHA256SUMS`;
do not expect a local rebuild to have the same digest.

See the [complete installer and repository index](https://github.com/insightos-community/quick-start/blob/main/README.build.md) for assembly order,
platform locks and end-to-end validation. Local build commands do not publish a
Release. Publishing requires repository write access and a new version tag;
existing release tags/assets should not be replaced.
# Windows x64 native Ability packages

Use Visual Studio 2022 x64 developer PowerShell, Git, authenticated GitHub CLI,
and uv 0.12.12. From a clean checkout, prepare the exact native dependencies:

```powershell
git clone https://github.com/insightos-community/ability-scaffold.git sources/scaffold
git -C sources/scaffold checkout 90917ecb0953605fa7bc47ea54e17a7e8273c1af
git clone https://github.com/insightos-community/Ability-SDK-Python.git sources/ability-sdk
git -C sources/ability-sdk checkout 064c36d0510b387318027213b39aaa2ffb7ea35e
./.github/scripts/windows-build.ps1
```

The script downloads and checks the locked Robot SDK release wheels, builds the
native `ability.exe` with static CRT, builds/installs the scaffold, Ability SDK
and R1 Pro project wheels, runs unit tests using CPython 3.13.15, then packages
all seven abilities. `.output/windows/` includes ZIPs, source/launcher records
and SHA-256 checksums. ZIPs retain `arch: x86_64` and contain `bin/ability.exe`.

The [Windows workflow](.github/workflows/windows.yml) runs this same script and
uploads development artifacts. Package validation does not establish full
AbilityFramework/Robot/MuJoCo lifecycle or physical GPU support; the explicit
MuJoCo product tests still require a running configured scene.
