import hashlib,json,subprocess
from pathlib import Path
pins=[('insightos-community/Ability-SDK-Python', 'v0.4.0-insightos.2026.2', '3a90b6d94ea6babb890ffcd285ba55e5fef6fc8c'), ('insightos-community/robot-sdk', 'v0.4.0-insightos.2026.2', '59a1a8364c3d0330f1802c020e37544e0dbfa7c5'), ('insightos-community/ability-scaffold', 'v1.2.0-insightos.2026.2', 'f85b9cce4f0040671fa74ebf0a587b7e4c65a7df')]
records=[]
for repo,tag,sha in pins:
 dest=Path('.output/deps')/repo.split('/')[-1];dest.mkdir(parents=True)
 subprocess.run(['gh','release','download',tag,'--repo',repo,'--dir',str(dest),'--pattern','*.whl','--pattern','SHA256SUMS','--pattern','release.json'],check=True)
 sums=dict(line.split('  ',1)[::-1] for line in (dest/'SHA256SUMS').read_text().splitlines())
 for p in dest.iterdir():
  if p.name!='SHA256SUMS' and hashlib.sha256(p.read_bytes()).hexdigest()!=sums[p.name]:raise SystemExit('Dependency checksum mismatch')
 meta=json.loads((dest/'release.json').read_text())
 if meta['tag']!=tag or meta['source_commit']!=sha:raise SystemExit('Dependency version mismatch')
 records.append(dict(repository=repo,tag=tag,source_commit=sha,checksums=sums))
Path('.output/dependencies.json').write_text(json.dumps(records,indent=2)+'\n')
