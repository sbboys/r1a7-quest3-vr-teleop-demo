#!/usr/bin/env python3
"""Check publication integrity without importing research or robot modules."""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'r1a7_threading_project/docs/source_records/2026-11-09_grasp_insertion'
X = ROOT / 'r1a7_threading_project/docs/experiments/2026-11-09_hole_unet'

def main():
    manifest = json.loads((A / 'manifest.json').read_text())
    compiled = 0
    for row in manifest['files']:
        p = ROOT / row['repository_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], p
        assert p.stat().st_size == row['size_bytes']
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', row['repository_path']], cwd=ROOT, stderr=subprocess.STDOUT)
        if p.suffix == '.py' or row['kind'] == 'before_change_backup':
            compile(p.read_bytes(), str(p), 'exec')
            compiled += 1
    for p in (A / 'research_code_snapshot').glob('*.diff.json'):
        json.loads(p.read_text())
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', str(p.relative_to(ROOT))], cwd=ROOT, stderr=subprocess.STDOUT)
    original = next(r for r in manifest['files'] if r['kind'] == 'original_document')
    with zipfile.ZipFile(ROOT / original['repository_path']) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns)) for p in root.findall('.//w:body//w:p', ns)]
        paragraphs = [p for p in paragraphs if p]
    text = html.unescape((A / 'worklog_fulltext_zh.md').read_text()).replace('<br>', '')
    assert all(p in text for p in paragraphs), 'source paragraph missing'
    assert len(manifest['success_snapshot_comparison']) == 4
    assert all(p['identical'] and p['current_sha256'] == p['snapshot_sha256'] for p in manifest['success_snapshot_comparison'])
    split = json.loads((X / 'splits_hole_board_grouped_v1.json').read_text())
    parts = [set(split[k]) for k in ['train', 'val', 'test']]
    assert list(map(len, parts)) == [24, 6, 6]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    qa = list(csv.DictReader((X / 'qa_v3/qa_mask_stats.csv').open()))
    assert len(qa) == 108
    config = json.loads((X / 'runs/hole_board_unet_v1/config.json').read_text())
    assert config['classes'] == {'0': 'background', '1': 'board', '2': 'hole'}
    history = list(csv.DictReader((X / 'runs/hole_board_unet_v1/history.csv').open()))
    assert len(history) == 40
    best = max(history, key=lambda r: float(r['val_score']))
    assert int(best['epoch']) == 38 and abs(float(best['val_hole_dice']) - 0.8557273610594628) < 1e-12
    assert json.loads((X / 'history_summary.json').read_text())['best_history_row'] == best
    assert len(manifest['external_checkpoints']) == 2
    # Existing console motion-output entry points are preserved.
    for rel in ['vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py',
                'vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py',
                'transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py']:
        p = 'r1a7_threading_project/' + rel
        assert (ROOT / p).read_bytes() == subprocess.check_output(['git', 'show', manifest['baseline_commit'] + ':' + p], cwd=ROOT)
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', '-z', manifest['baseline_commit']], cwd=ROOT).decode().split('\0'))
    changed |= set(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0'))
    links = fences = 0
    for rel in changed - {''}:
        p = ROOT / rel
        if not p.is_file():
            continue
        assert p.stat().st_size <= 20 * 1024 * 1024, rel
        if p.suffix == '.md':
            content = p.read_text()
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
                target = target.split('#')[0].strip('<>')
                if not target or re.match(r'^[a-z]+:', target):
                    continue
                assert (p.parent / urllib.parse.unquote(target)).exists(), (rel, target)
                links += 1
            if p.name == 'r1a7_grasp_insertion_sop_2026-11-09_zh.md':
                for code in re.findall(r'```bash\n(.*?)```', content, re.S):
                    subprocess.run(['bash', '-n'], input=code, text=True, check=True)
                    fences += 1
    report = {'archive_date': manifest['archive_date'], 'status': 'PASS', 'source_files_hash_verified': len(manifest['files']),
              'source_python_and_backups_compiled_without_import': compiled, 'source_paragraphs_covered': len(paragraphs),
              'success_snapshot_pairs_identical': 4, 'hole_split_groups': [24, 6, 6], 'qa_rows': len(qa),
              'training_history_rows': len(history), 'maximum_val_score_epoch': int(best['epoch']),
              'local_links_checked': links, 'sop_bash_blocks_syntax_only': fences,
              'active_console_control_files_unchanged': True, 'runtime_or_hardware_executed': False}
    (X / 'archive_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
