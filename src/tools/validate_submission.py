# tools/validate_submission.py
"""
Validate submission.json (array of objects with image_id & predicted_class)
Usage:
 python tools/validate_submission.py submission.json data/test_public checkpoints/class_mapping.json
"""
import os
import sys
import json

def validate(sub_path, test_dir, mapping_path):
    with open(sub_path, 'r') as f:
        subs = json.load(f)
    if not isinstance(subs, list):
        print("Submission must be a list of objects.")
        return False
    # gather image ids
    sub_ids = [s.get('image_id') for s in subs]
    if None in sub_ids:
        print("Each submission entry must have 'image_id' field.")
        return False
    sub_set = set(sub_ids)
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    test_set = set(test_files)
    missing = test_set - sub_set
    extra = sub_set - test_set
    if missing:
        print(f"Missing {len(missing)} images in submission. Example: {list(missing)[:5]}")
        return False
    if extra:
        print(f"Submission contains {len(extra)} unknown image ids. Example: {list(extra)[:5]}")
        return False
    # load allowed labels
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    allowed = set(mapping.values()) if isinstance(mapping, dict) else set(mapping)
    invalid = [s for s in subs if s['predicted_class'] not in allowed]
    if invalid:
        print(f"Found {len(invalid)} invalid predicted_class values. Example: {invalid[:3]}")
        return False
    print("Submission format OK.")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python tools/validate_submission.py submission.json test_dir mapping.json")
        sys.exit(2)
    ok = validate(sys.argv[1], sys.argv[2], sys.argv[3])
    if not ok:
        sys.exit(1)
