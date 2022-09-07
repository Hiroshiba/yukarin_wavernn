# TrainerのpthファイルからPredictorを抜き出す。output名がなければ、そのイテレーションのpredictorとして保存される。

import argparse
from pathlib import Path
from typing import Optional

import torch
import yaml
from yukarin_wavernn.network.wave_rnn import WaveRNN
from yukarin_wavernn.trainer import create_trainer


def extract_predictor(trained_dir: Path, output: Optional[Path]):
    config_dict = yaml.safe_load((trained_dir / "config.yaml").read_text())
    trainer = create_trainer(config_dict=config_dict, output=trained_dir)

    trainer_path = trained_dir.glob("trainer_*.pth").__next__()
    trainer.load_state_dict(torch.load(trainer_path, map_location="cpu"))

    if output is None:
        trainer_iteration = int(trainer_path.stem.split("_")[-1])
        output = trained_dir / f"predictor_{trainer_iteration}.pth"

    predictor: WaveRNN = trainer.updater.get_all_models()["main"].predictor
    torch.save(predictor.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    extract_predictor(trained_dir=args.trained_dir, output=args.output)
