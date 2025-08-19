from dataclasses import dataclass, field

def to_obj(obj, cls):
    return cls(**obj) if not isinstance(obj, cls) else obj
# ---------------------------
# Config
# ---------------------------
@dataclass
class SaveLogProbsCfg:
    mode: str = "topk"  # "none" | "full" | "topk"
    top_k: int = 64      # used if mode == "topk"
    dtype: str = "float16"  # storage precision for probs/logits: float16 | bfloat16 | float32

@dataclass
class GenCfg:
    name: str = "GE1"
    seed: int = 42
    device: str = "auto"      # "auto" or explicit like "cuda:0"
    dtype: str = "bfloat16"    # model weights dtype: float16|bfloat16|float32
    max_len: int = 2048
    jacobi_tokens: int = 100
    batch_size: int = 1        # generation currently streams per sample; keep 1 for correctness
    do_sample: bool = True
    temperature: float = 0.7
    save_hidden_dtype: str = "float16"  # storage precision for hidden states
    save_every: int = 2000      # write after N samples to a new shard
    output_folder: str = "./datasets"

@dataclass
class DataCfg:
    data_name: str = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    data_files: str = "ShareGPT_V3_unfiltered_cleaned_split.json"
    split: str = "train"
    start: int = 0
    end: int = 20000
    adapter: str = "sharegpt"  # which dataset adapter to use

@dataclass
class ModelCfg:
    model_path: str = "Qwen"
    model_name: str = "Qwen2.5-0.5B-Instruct"

@dataclass
class GenDataConfig:
    gen: GenCfg
    data: DataCfg
    model: ModelCfg
    save_probs: SaveLogProbsCfg

@dataclass
class TMetaCfg:
    project: str = "Jacobi-test-3"
    name: str = "CFG11"
    api_key: str = ""
    debug_mode: bool = False
    cpdir: str = "./jacobi_test_weights"

@dataclass
class TModelCfg:
    basepath: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_jacobi_tokens: int = 3
    num_prev_sequences: int = 2
    adapter_insertion_freq: int = 4 
    adapter_type: str = "Qwen2MLP"
    shared_adapter: bool = False
    fuse_prev_hidden_states: bool = True
    shared_jacobi_token: bool = True
    jacobi_adapter_kwargs: dict =field(
        default_factory=lambda: {
            "intermediate_ratio": None,
            "clamp": False
        }
    )
    use_pre_layer_norm: bool = True
    token_sets_inline: bool = False

@dataclass
class TDataCfg:
    tr_path: str = "./data_root/ShareGPT_Vicuna_unfiltered_Qwen2.5-0.5B-Instruct"
    te_path: str = "./data_root/ShareGPT_Vicuna_unfiltered_Qwen2.5-0.5B-Instruct"
    schedule: str = "tail"
    train_data_portion: float = 0.95 # :p
    test_data_portion: float = 0.95  # p:
    pad_token_id: int = 151643
    max_len: int = 2048
    num_workers: int = 2

@dataclass
class TrainCfg:
    mixed_precision: str = "bf16"
    bs: int = 1
    lr: float = 4.0e-4
    loss_method: str = "SmoothL1Loss"
    gamma: float = None
    do_clip: bool = False
    clip_max:float = 1.0e2
    clip_min:float = -1.0e2
    initialise_method: str = "kaiming"
    is_warmup: bool = True
    num_epochs: int = 20
    num_warmup_steps: int = 1
    total_steps: int = 518000
    jcb_weight_reg: bool = False
    adp_weight_reg: bool = False
    jratio: float = 0.1
    aratio: float = 1.0e-5
    pratio: float = 0.2
    vratio: float = 1.0
    gradient_accumulation_steps: int = 1
    grad_clip: float = 0.5
    b1: float = 0.9
    b2: float = 0.95
    statepath: str = None
    save_freq: int = 1

@dataclass
class TrainScriptConfig:
    meta: TMetaCfg
    model: TModelCfg
    data: TDataCfg
    train: TrainCfg