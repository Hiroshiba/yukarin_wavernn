from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from yukarin_wavernn.utility import dataclass_utility
from yukarin_wavernn.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    sampling_rate: int
    sampling_length: int
    input_wave_glob: str
    input_silence_glob: str
    input_local_glob: str
    bit_size: int
    gaussian_noise_sigma: float
    mulaw: bool
    local_sampling_rate: Optional[int]
    local_padding_size: int
    speaker_dict_path: Optional[str]
    num_speaker: Optional[int]
    seed: int
    num_train: Optional[int]
    num_test: int
    num_times_evaluate: int
    time_length_evaluate: float
    local_padding_time_length_evaluate: float


class LocalNetworkType(str, Enum):
    gru = "gru"
    skip_dilated_cnn = "skip_dilated_cnn"


@dataclass
class NetworkConfig:
    bit_size: int
    hidden_size: int
    local_size: int
    conditioning_size: int
    embedding_size: int
    linear_hidden_size: int
    local_scale: int
    local_layer_num: int
    local_network_type: str
    speaker_size: int
    speaker_embedding_size: int


@dataclass
class LossConfig:
    eliminate_silence: bool
    mean_silence: bool


@dataclass
class TrainConfig:
    batchsize: int
    eval_batchsize: Optional[int]
    log_iteration: int
    eval_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    num_processes: Optional[int] = None
    use_amp: bool = False
    use_multithread: bool = False
    linear_shift: Optional[Dict[str, Any]] = None
    step_shift: Optional[Dict[str, Any]] = None
    optuna: Optional[Dict[str, Any]] = None


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    loss: LossConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict):
    if "gaussian_noise_sigma" not in d["dataset"]:
        d["dataset"]["gaussian_noise_sigma"] = 0.0

    if "mulaw" not in d["dataset"]:
        d["dataset"]["mulaw"] = False

    if "eliminate_silence" not in d["loss"]:
        d["loss"]["eliminate_silence"] = True

    if "local_padding_size" not in d["dataset"]:
        d["dataset"]["local_padding_size"] = 0

    if "num_train" not in d["dataset"]:
        d["dataset"]["num_train"] = None

    if "mean_silence" not in d["loss"]:
        d["loss"]["mean_silence"] = True

    if "speaker_size" not in d["network"]:
        d["network"]["speaker_size"] = 0

    if "speaker_embedding_size" not in d["network"]:
        d["network"]["speaker_embedding_size"] = 0

    if "speaker_dict_path" not in d["dataset"]:
        d["dataset"]["speaker_dict_path"] = None

    if "num_times_evaluate" not in d["dataset"]:
        d["dataset"]["num_times_evaluate"] = None

    if "time_length_evaluate" not in d["dataset"]:
        d["dataset"]["time_length_evaluate"] = None

    if "local_padding_time_length_evaluate" not in d["dataset"]:
        d["dataset"]["local_padding_time_length_evaluate"] = 0

    if "local_sampling_rate" not in d["dataset"]:
        d["dataset"]["local_sampling_rate"] = None

    if "eval_batchsize" not in d["train"]:
        d["train"]["eval_batchsize"] = None

    if "local_network_type" not in d["network"]:
        d["network"]["local_network_type"] = "gru"


def assert_config(config: Config):
    assert config.dataset.bit_size == config.network.bit_size

    if config.dataset.speaker_dict_path is not None:
        assert config.dataset.num_speaker == config.network.speaker_size
