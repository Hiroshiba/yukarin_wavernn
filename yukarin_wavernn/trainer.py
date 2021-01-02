import warnings
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import Trainer, extensions, triggers
from pytorch_trainer.training.updaters import StandardUpdater
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.optimizer import Optimizer

from yukarin_wavernn.config import Config, assert_config
from yukarin_wavernn.dataset import create as create_dataset
from yukarin_wavernn.evaluator import GenerateEvaluator
from yukarin_wavernn.generator import Generator
from yukarin_wavernn.model import Model, create_predictor
from yukarin_wavernn.utility.pytorch_utility import init_weights
from yukarin_wavernn.utility.trainer_extension import TensorboardReport, WandbReport
from yukarin_wavernn.utility.trainer_utility import LowValueTrigger, create_iterator


def create_trainer(
    config_dict: Dict[str, Any],
    output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()
    assert_config(config)

    output.mkdir(exist_ok=True, parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    predictor = create_predictor(config.network)
    model = Model(
        loss_config=config.loss,
        predictor=predictor,
        local_padding_size=config.dataset.local_padding_size,
    )
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)

    device = torch.device("cuda")
    model.to(device)

    # dataset
    _create_iterator = partial(
        create_iterator,
        batch_size=config.train.batchsize,
        num_processes=config.train.num_processes,
        use_multithread=config.train.use_multithread,
    )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets["train"], for_train=True)
    test_iter = _create_iterator(datasets["test"], for_train=False)
    eval_iter = _create_iterator(datasets["eval"], for_train=False)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    # optimizer
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop("name").lower()

    optimizer: Optimizer
    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    else:
        raise ValueError(n)

    # updater
    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    # trainer
    trigger_log = (config.train.log_iteration, "iteration")
    trigger_eval = (config.train.eval_iteration, "iteration")
    trigger_stop = (
        (config.train.stop_iteration, "iteration")
        if config.train.stop_iteration is not None
        else None
    )

    trainer = Trainer(updater, stop_trigger=trigger_stop, out=output)

    shift_ext = None
    if config.train.linear_shift is not None:
        shift_ext = extensions.LinearShift(**config.train.linear_shift)
    if config.train.step_shift is not None:
        shift_ext = extensions.StepShift(**config.train.step_shift)
    if shift_ext is not None:
        trainer.extend(shift_ext)

    ext = extensions.Evaluator(test_iter, model, device=device)
    trainer.extend(ext, name="test", trigger=trigger_log)

    generator = Generator(
        config=config,
        predictor=predictor,
        use_gpu=True,
    )
    generate_evaluator = GenerateEvaluator(
        generator=generator,
        time_length=config.dataset.time_length_evaluate,
        local_padding_time_length=config.dataset.local_padding_time_length_evaluate,
    )
    ext = extensions.Evaluator(eval_iter, generate_evaluator, device=device)
    trainer.extend(ext, name="eval", trigger=trigger_eval)

    ext = extensions.snapshot_object(
        predictor,
        filename="predictor_{.updater.iteration}.pth",
        n_retains=5,
    )
    trainer.extend(
        ext,
        trigger=LowValueTrigger("eval/main/mcd", trigger=trigger_eval),
    )

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.observe_lr(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(["iteration", "main/loss", "test/main/loss"]),
        trigger=trigger_log,
    )
    trainer.extend(
        TensorboardReport(writer=SummaryWriter(Path(output))), trigger=trigger_log
    )

    if config.project.category is not None:
        ext = WandbReport(
            config_dict=config.to_dict(),
            project_category=config.project.category,
            project_name=config.project.name,
            output_dir=output.joinpath("wandb"),
        )
        trainer.extend(ext, trigger=trigger_log)

    (output / "struct.txt").write_text(repr(model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    ext = extensions.snapshot_object(
        trainer,
        filename="trainer_{.updater.iteration}.pth",
        n_retains=1,
        autoload=True,
    )
    trainer.extend(ext, trigger=trigger_eval)

    return trainer
