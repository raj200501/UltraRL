from pathlib import Path

import pandas as pd

from ultrarl.run import main

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main.yaml"


def test_smoke_pipeline_end_to_end(tmp_path, capsys):
    out = tmp_path / "smoke"
    main(["--config", str(CONFIG), "--out", str(out), "--smoke", "--workers", "1"])
    summary = pd.read_csv(out / "eval_summary.csv")
    per_seed = pd.read_csv(out / "eval_per_seed.csv")
    assert set(summary["setting"]) == {"train_like", "heldout_long"}
    assert len(summary) == 2 * (11 + 6)  # settings x (methods + final-checkpoint variants)
    assert per_seed["success"].between(0, 1).all()
    assert len(list((out / "checkpoints").glob("*.pt"))) == 2 * 6
    assert len(list((out / "curves").glob("*.csv"))) == 6
    # Resuming skips already-trained runs and reproduces the evaluation exactly.
    capsys.readouterr()
    main(["--config", str(CONFIG), "--out", str(out), "--smoke", "--workers", "1"])
    assert "[train] 0 runs to train" in capsys.readouterr().out
    pd.testing.assert_frame_equal(per_seed, pd.read_csv(out / "eval_per_seed.csv"))
