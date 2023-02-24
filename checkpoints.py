from typing import Optional

import os

def find_checkpoint(checkpoints_dir: str, load_latest: bool) -> Optional[str]:
    if load_latest:
        checkpoint_path = os.path.join(checkpoints_dir, 'muzero_latest.ckpt')
        return checkpoint_path

    max_score = None
    max_score_fn = None
    for fn in os.listdir(checkpoints_dir):
        if not fn.endswith('.ckpt'):
            continue
        if not fn.startswith('muzero_best_'):
            continue

        filename = os.path.splitext(fn)[0]
        score_str = filename.split('_')[-1]
        score = float(score_str)
        if max_score is None or score > max_score:
            max_score = score
            max_score_fn = fn

    if max_score_fn is not None:
        checkpoint_path = os.path.join(checkpoints_dir, max_score_fn)
        return checkpoint_path

    return None
