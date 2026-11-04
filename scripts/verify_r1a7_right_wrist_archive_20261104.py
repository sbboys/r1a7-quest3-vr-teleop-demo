#!/usr/bin/env python3
"""Verify the 2026-11-03 research archive without importing models or hardware code."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import html
import io
import json
import re
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'r1a7_threading_project/docs/source_records/2026-11-03_right_wrist_open'
RESULTS = ROOT / 'r1a7_threading_project/docs/experiments/2026-11-03_right_wrist_open'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    compiled = 0
    for record in manifest['files']:
        path = ROOT / record['repository_path']
        assert path.stat().st_size == record['size_bytes'], path
        assert digest(path) == record['sha256'], path
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', record['repository_path']], cwd=ROOT, stderr=subprocess.STDOUT)
        # Backups retain their original names and exact bytes.
        if path.suffix == '.py' or record['kind'] == 'historical_source_backup':
            compile(path.read_bytes(), str(path), 'exec')
            compiled += 1
    source = next(x for x in manifest['files'] if x['kind'] == 'source_document')
    transcript = html.unescape((ARCHIVE / 'worklog_fulltext_zh.md').read_text()).replace('<br>', '')
    with zipfile.ZipFile(ROOT / source['repository_path']) as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
                      for p in document.findall('.//w:body//w:p', ns)]
        paragraphs = [p for p in paragraphs if p]
        assert all(p in transcript for p in paragraphs), 'fulltext omitted source text'
    split = json.loads((RESULTS / 'splits_right_wrist_open_grouped_v1.json').read_text())
    parts = {k: set(split[k]) for k in ['train', 'val', 'test']}
    assert [len(parts[k]) for k in parts] == [47, 8, 9]
    assert not (parts['train'] & parts['val'] or parts['train'] & parts['test'] or parts['val'] & parts['test'])
    qa = list(csv.DictReader(io.StringIO((RESULTS / 'qa_mask_stats.csv').read_text())))
    assert len(qa) == 64
    assert {r['name'].removesuffix('.png') for r in qa} == set.union(*parts.values())
    assert all(int(r['width']) == 1280 and int(r['height']) == 800 and int(r['ignore']) == 0 for r in qa)
    assert min(int(r['tool']) for r in qa) == 1878
    metrics = json.loads((RESULTS / 'eval_v2_v3_test/metrics.json').read_text())
    assert set(metrics['test_ids']) == parts['test']
    assert len(metrics['per_image']) == 9
    assert abs(metrics['v2']['tool']['dice'] - 0.5691879063048189) < 1e-12
    assert abs(metrics['v3']['tool']['dice'] - 0.8915323564178096) < 1e-12
    history = json.loads((RESULTS / 'runs/v3_mixed_replay/history.json').read_text())
    best = json.loads((RESULTS / 'runs/v3_mixed_replay/best_metrics.json').read_text())
    assert len(history) == 20 and [history[0]['epoch'], history[-1]['epoch']] == [73, 92]
    assert all(r['old_guard_ok'] for r in history)
    assert best['epoch'] == max(history, key=lambda r: r['combined_score'])['epoch'] == 90
    assert manifest['checkpoints'][0]['sha256'] == manifest['checkpoints'][1]['sha256']
    assert manifest['checkpoints'][0]['sha256'] != manifest['checkpoints'][2]['sha256']
    counts = Counter(str(Path(x['relative_to_dataset']).parent) for x in manifest['external_assets'])
    assert len(manifest['external_assets']) == 205
    assert counts['images/right_wrist'] == counts['masks/right_wrist'] == 64
    assert counts['qa_contact_sheets'] == 4 and counts['eval_v2_v3_test'] == 9
    controller = next(x for x in manifest['files'] if x['kind'] == 'archive_only_controller')
    tree = ast.parse((ROOT / controller['repository_path']).read_text())
    conditions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'wrist_close', 'ext_close'}:
                    conditions[target.id] = ast.unparse(node.value)
    assert 'ev' not in conditions['wrist_close'] and 'eu' in conditions['wrist_close'] and 'jitter' in conditions['wrist_close']
    assert all(k in conditions['ext_close'] for k in ['eu', 'ev', 'jitter'])
    diffs = list((ARCHIVE / 'research_code_snapshot').glob('*.diff.json'))
    assert len(diffs) == 3
    for diff_path in diffs:
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', str(diff_path.relative_to(ROOT))], cwd=ROOT, stderr=subprocess.STDOUT)
    delta = json.loads((ARCHIVE / 'research_code_snapshot/controller_since_before_remove_wrist_ev.diff.json').read_text())
    removed = [l for l in delta['patch'].splitlines() if l.startswith('-') and not l.startswith('---')]
    added = [l for l in delta['patch'].splitlines() if l.startswith('+') and not l.startswith('+++')]
    assert len(removed) == 8 and not added
    assert all(d['identical'] and d['source_sha256'] == d['repository_sha256'] for d in manifest['checked_dependencies'])
    # Confirm original integrated hardware entry points were not changed.
    for rel in ['vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py',
                'vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py',
                'transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py']:
        repo_path = 'r1a7_threading_project/' + rel
        assert (ROOT / repo_path).read_bytes() == subprocess.check_output(['git', 'show', manifest['baseline_commit'] + ':' + repo_path], cwd=ROOT)
    paths = set(subprocess.check_output(['git', 'diff', '--name-only', '-z', manifest['baseline_commit']], cwd=ROOT).decode().split('\0'))
    paths |= set(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0'))
    link_count = 0
    for rel in paths - {''}:
        path = ROOT / rel
        if not path.is_file():
            continue
        assert path.stat().st_size <= 20 * 1024 * 1024, rel
        if path.suffix == '.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                target = target.split('#')[0].strip('<>')
                if not target or re.match(r'^[a-z]+:', target):
                    continue
                assert (path.parent / urllib.parse.unquote(target)).exists(), (rel, target)
                link_count += 1
    report = {'archive_date': manifest['archive_date'], 'manifest_files_verified': len(manifest['files']),
              'source_python_files_compiled_without_import': compiled, 'source_paragraphs_covered': len(paragraphs),
              'split_counts': {k: len(v) for k, v in parts.items()}, 'qa_rows': len(qa),
              'external_image_manifest_count': len(manifest['external_assets']), 'local_links_checked': link_count,
              'historical_metrics_and_gate_diff_verified': True, 'active_control_files_unchanged': True,
              'runtime_or_hardware_executed': False, 'status': 'PASS'}
    (RESULTS / 'archive_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
