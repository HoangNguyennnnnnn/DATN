"""Create non-overlapping train/test identity lists for FaceVerse and FaceScape.

This script only creates ID text files and never edits dataset files.
"""

import os
import re


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACEVERSE_ROOT = "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse"
FACESCAPE_ROOT = "/mnt/16TData/Datasets/FaceScape"


def _norm_id(token: str):
    token = str(token).strip()
    if not token:
        return None
    if token.isdigit():
        return str(int(token))
    return token


def _sort_key(token: str):
    if token.isdigit():
        return (0, int(token))
    return (1, token)


def _write_ids(path: str, ids):
    with open(path, "w", encoding="utf-8") as f:
        for token in ids:
            f.write(f"{token}\n")


def _scan_faceverse_ids(root_dir: str):
    ids = set()
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"FaceVerse root not found: {root_dir}")

    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if not os.path.isdir(full):
            continue
        # Expected folder name: 001_01
        token = entry.split("_", 1)[0]
        token = _norm_id(token)
        if token is not None:
            ids.add(token)
    return sorted(ids, key=_sort_key)


def _scan_facescape_ids(root_dir: str):
    ids = set()
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"FaceScape root not found: {root_dir}")

    # Nested mode: root/<id>/models_reg/*.obj
    subdirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    for sub in subdirs:
        if sub.isdigit():
            ids.add(_norm_id(sub))

    # Flat mode fallback: root/*.obj like 001_01.obj
    for name in sorted(os.listdir(root_dir)):
        if not name.endswith(".obj"):
            continue
        stem = os.path.splitext(name)[0]
        match = re.match(r"^(\d+)", stem)
        if match:
            ids.add(_norm_id(match.group(1)))

    ids.discard(None)
    return sorted(ids, key=_sort_key)


def main():
    faceverse_ids = _scan_faceverse_ids(FACEVERSE_ROOT)
    facescape_ids = _scan_facescape_ids(FACESCAPE_ROOT)

    fv_test = faceverse_ids[:10]
    fv_train = [x for x in faceverse_ids if x not in fv_test]

    fs_test = facescape_ids[:10]
    fs_train = [x for x in facescape_ids if x not in fs_test]

    _write_ids(os.path.join(PROJECT_ROOT, "test_faceverse_ids.txt"), fv_test)
    _write_ids(os.path.join(PROJECT_ROOT, "train_faceverse_ids.txt"), fv_train)
    _write_ids(os.path.join(PROJECT_ROOT, "test_facescape_ids.txt"), fs_test)
    _write_ids(os.path.join(PROJECT_ROOT, "train_facescape_ids.txt"), fs_train)

    print(f"FaceVerse IDs: total={len(faceverse_ids)} train={len(fv_train)} test={len(fv_test)}")
    print(f"FaceScape IDs: total={len(facescape_ids)} train={len(fs_train)} test={len(fs_test)}")
    print("Done: wrote train/test ID files (non-overlap by construction).")


if __name__ == "__main__":
    main()
