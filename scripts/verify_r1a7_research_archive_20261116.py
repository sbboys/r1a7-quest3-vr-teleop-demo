#!/usr/bin/env python3
"""Verify archived evidence without importing models, cameras or control code."""
from pathlib import Path
import hashlib
import html
import json
import re
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'r1a7_threading_project/docs/source_records/2026-11-16_research'
X = ROOT / 'r1a7_threading_project/docs/experiments/2026-11-16_research'

def load(rel):
    return json.loads((X / rel).read_text())

def main():
    manifest = json.loads((A / 'manifest.json').read_text())
    compiled = 0
    for record in manifest['files']:
        p = ROOT / record['repository_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256'], p
        assert p.stat().st_size == record['size_bytes']
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', record['repository_path']], cwd=ROOT, stderr=subprocess.STDOUT)
        if p.suffix == '.py' or record['kind'] == 'before_change_backup':
            compile(p.read_bytes(), str(p), 'exec')
            compiled += 1
    paragraphs = 0
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    for doc in manifest['documents']:
        p = ROOT / doc['repository_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == doc['sha256']
        transcript = html.unescape((ROOT / doc['transcript']).read_text()).replace('<br>', '')
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.startswith('word/media/'):
                    assert (A / 'assets' / Path(name).name).read_bytes() == z.read(name)
            root = ET.fromstring(z.read('word/document.xml'))
            texts = [''.join(t.text or '' for t in p.findall('.//w:t', ns)) for p in root.findall('.//w:body//w:p', ns)]
            for text in (t.rstrip() for t in texts if t.strip()):
                assert text in transcript, (doc['source_path'], text[:60])
                paragraphs += 1
    for p in (A / 'research_code_snapshot').glob('*.diff.json'):
        json.loads(p.read_text())
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', str(p.relative_to(ROOT))], cwd=ROOT, stderr=subprocess.STDOUT)
    old = load('old_joint/training_runtime/splits_insertion_joint_grouped_seed44_v1.json')
    new = load('new_board/training/wire_v5_newboard/splits/newboard_grouped_timeblock_v1.json')
    new_hole = load('new_board/training/hole_v3_newboard/splits/newboard_grouped_timeblock_v1.json')
    for split, counts in [(old, [25, 5, 5]), (new, [15, 3, 3])]:
        sets = [set(split[k]) for k in ['train', 'val', 'test']]
        assert list(map(len, sets)) == counts
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert all(new[k] == new_hole[k] for k in ['train', 'val', 'test'])
    wire = load('new_board/training/wire_v5_newboard/runs/wire_v5_newboard_pilot01/newboard_test_metrics.json')
    hole = load('new_board/training/hole_v3_newboard/runs/hole_v3_newboard_pilot01/test_metrics.json')
    assert wire['best_epoch'] == 123 and abs(wire['newboard_test']['mean_fg_dice'] - 0.9026257394079478) < 1e-12
    assert hole['best_epoch'] == 64 and hole['test_metrics']['per_camera']['external']['hole_dice'] == 0
    assert abs(hole['test_metrics']['score'] - 0.3756477975843484) < 1e-12
    assert load('new_board/annotation_audit/annotation_audit_v2.json')['audit_pass'] is True
    raw = load('old_joint/raw_session_summary/session_summary.json')
    assert (raw['saved_records'], raw['saved_images'], raw['dropped_queue']) == (197, 591, 0)
    old_hole = load('old_joint/frozen_results/hole_v2_epoch52_reserved_test.json')['old_hole_test']['v2']
    assert abs(old_hole['mean_fg_dice'] - 0.8586391076131592) < 1e-12
    assert abs(old_hole['score'] - 0.8256479275856521) < 1e-12
    visual = ROOT / 'r1a7_threading_project/vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py'
    assert 'wire_v5_newboard_seed44_epoch123.pt' in visual.read_text()
    assert 'hole_v3_newboard_seed44_epoch64.pt' in visual.read_text()
    assert len(manifest['external_rgb_masks']) == 504
    expected = ['31dcc17b4353be524af6bd75613ecf8802745ad0a2f07fdc1bb19d0587be88d7',
                '467c57b3e74f9b885013bc8c0a77a3881fad53283d0325b2e0ba3f450d0a6b52',
                'c72a9f40c4890094a33db2ed3f50da54e5e7ce5ebb11c56ed906d6b3d174a4e7',
                '4ec3923e49e12925520985b8b331c465d24c44bf14dce89107984fd9d6685763',
                '9ad042afbb39543ec8deaa1c5d80512f8335fb2bbe6e9c18b237b82540334321',
                '462288ecc3ba00c1f6c63d6b6cbd3f9eab120a1b98135966b531a435fa95c9f4']
    assert [x['sha256'] for x in manifest['external_packages_models']] == expected
    for rel in ['vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py',
                'vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py',
                'transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py']:
        p = 'r1a7_threading_project/' + rel
        assert (ROOT / p).read_bytes() == subprocess.check_output(['git', 'show', manifest['baseline_commit'] + ':' + p], cwd=ROOT)
    paths = set(subprocess.check_output(['git', 'diff', '--name-only', '-z', manifest['baseline_commit']], cwd=ROOT).decode().split('\0'))
    paths |= set(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0'))
    links = blocks = 0
    for rel in paths - {''}:
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
            if p.name == 'r1a7_research_sop_2026-11-16_zh.md':
                for code in re.findall(r'```bash\n(.*?)```', content, re.S):
                    subprocess.run(['bash', '-n'], input=code, text=True, check=True)
                    blocks += 1
    result = {'archive_date': manifest['archive_date'], 'status': 'PASS', 'source_file_hashes': len(manifest['files']),
              'documents_verified': len(manifest['documents']), 'source_paragraphs_covered': paragraphs,
              'python_sources_and_backups_compiled_without_import': compiled,
              'old_joint_split_groups': [25, 5, 5], 'newboard_split_groups': [15, 3, 3],
              'model_and_GT_package_hashes': 6, 'external_rgb_mask_manifest_entries': 504,
              'local_links_checked': links, 'sop_shell_blocks_syntax_only': blocks,
              'active_console_output_chain_unchanged': True, 'runtime_or_hardware_executed': False}
    (X / 'archive_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
