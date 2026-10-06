"""Central configuration for the CorpPolicyLM pipeline (Assignment 1A).

Every script imports paths and hyper-parameters from here so the notebook,
the CLI scripts and the tests all use the same values. Any value can be
overridden with an environment variable of the same name (handy on Colab).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return default if value is None else cast(value)


# ---------------------------------------------------------------- paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
SOURCES_CSV = DATA_DIR / "sources.csv"
EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
DOMAIN_CORPUS_DIR = DATA_DIR / "domain_corpus"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
INSTRUCTION_DIR = DATA_DIR / "instruction"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUTS_DIR / "reports"
FIGURE_DIR = OUTPUTS_DIR / "figures"
EVAL_DIR = OUTPUTS_DIR / "evaluations"
LOG_DIR = OUTPUTS_DIR / "logs"

MODELS_DIR = Path(_env("MODELS_DIR", str(PROJECT_ROOT / "models")))
CPT_MODEL_DIR = MODELS_DIR / "cpt"
ADAPTER_DIR = MODELS_DIR / "adapters"

# ---------------------------------------------------------------- model
# HR Policy & Corporate Documents -> T4 Choice 2 in the assignment table.
# The base (non-chat) checkpoint is used for CPT; never mix tokenizers.
MODEL_ID = _env("MODEL_ID", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# TinyLlama has a 2048-token context window; packed blocks match it.
BLOCK_SIZE = _env("BLOCK_SIZE", 2048, int)

# ---------------------------------------------------------------- cleaning
MIN_WORDS = _env("MIN_WORDS", 100, int)
NEAR_DUP_THRESHOLD = _env("NEAR_DUP_THRESHOLD", 0.8, float)  # shingle Jaccard
EVAL_FRACTION = _env("EVAL_FRACTION", 0.1, float)    # last 10% of every document -> eval
UNSEEN_FRACTION = _env("UNSEEN_FRACTION", 0.05, float)  # whole documents never seen -> eval_unseen
# Documents the probe prompts ask about (kept in the CPT training split).
PROBE_DOCUMENTS = ["Leave_Policy.txt", "travel_policy.txt", "mygov_travel_policy_dup.txt", "exit_policy.txt"]
SEED = _env("SEED", 42, int)

# ---------------------------------------------------------------- CPT
CPT = {
    "learning_rate": _env("CPT_LR", 5e-5, float),
    "num_train_epochs": _env("CPT_EPOCHS", 2, float),
    "max_steps": _env("CPT_MAX_STEPS", -1, int),
    "per_device_train_batch_size": _env("CPT_BATCH", 1, int),
    "gradient_accumulation_steps": _env("CPT_GRAD_ACCUM", 8, int),
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "max_grad_norm": 1.0,
}

# ---------------------------------------------------------------- QLoRA
# Adapter B (Balanced) from the assignment table.
ADAPTERS = {
    "A": {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]},
    "B": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]},
    "C": {"r": 32, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj", "o_proj"]},
}
ADAPTER_NAME = _env("ADAPTER_NAME", "B")
SFT = {
    "learning_rate": _env("SFT_LR", 2e-4, float),
    "num_train_epochs": _env("SFT_EPOCHS", 2, float),
    "max_steps": _env("SFT_MAX_STEPS", -1, int),
    "per_device_train_batch_size": _env("SFT_BATCH", 4, int),
    "gradient_accumulation_steps": _env("SFT_GRAD_ACCUM", 4, int),
    "max_length": _env("SFT_MAX_LEN", 512, int),
    "lora_dropout": 0.05,
    "warmup_ratio": 0.05,
    "logging_steps": 5,
}
INSTRUCTION_TRAIN_FRACTION = 0.8

# TinyLlama base ships without a chat template. We attach the official
# TinyLlama-1.1B-Chat-v1.0 (Zephyr-style) template so SFT and inference use
# exactly the format the TinyLlama family was chat-tuned with.
CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)
SYSTEM_PROMPT = (
    "You are CorpPolicyLM, an HR policy assistant. Answer using only the "
    "organisation's HR and corporate policy documents. Be concise and factual."
)

# ---------------------------------------------------------------- prompts
DOMAIN_PROMPTS = [
    "Under the MyGov Leave Policy, resources are eligible to avail during the probation only",
    "According to the MyGov Domestic Travel Policy, a resource who drives his own vehicle can claim expenses @",
    "As per the MyGov Exit Policy, either party may terminate the Agreement by giving a written notice of",
]
# Same three topics phrased as questions for the instruction-tuned model (Part B3).
DOMAIN_QUESTIONS = [
    "Under the MyGov Leave Policy, what leave can a resource avail during probation?",
    "Under the MyGov Domestic Travel Policy, how much can a resource claim for driving his own vehicle?",
    "Under the MyGov Exit Policy, how many days of written notice are needed to terminate the Agreement?",
]
# Reference answers taken verbatim from the MyGov policies (used to score outputs).
DOMAIN_REFERENCE = [
    ["casual", "sick"],
    ["rs.10", "rs. 10", "10 per km", "₹10"],
    # Exit notice depends on tenure: 60 days after 3 years' service, 90 days after 5 years.
    ["60 days", "90 days", "sixty", "ninety"],
]
GENERAL_PROMPTS = [
    ("The capital of France is", ["paris"]),
    ("Water boils at", ["100", "212"]),
    ("The speed of light is approximately", ["300,000", "300000", "3 x 10", "3×10", "3 × 10", "186,000", "299,792", "299792"]),
]
GEN_MAX_NEW_TOKENS = _env("GEN_MAX_NEW_TOKENS", 60, int)


def ensure_dirs():
    for path in [RAW_PDF_DIR, EXTRACTED_TEXT_DIR, DOMAIN_CORPUS_DIR, SPLITS_DIR,
                 PROCESSED_DIR, INSTRUCTION_DIR, REPORT_DIR, FIGURE_DIR, EVAL_DIR,
                 LOG_DIR, CPT_MODEL_DIR, ADAPTER_DIR]:
        path.mkdir(parents=True, exist_ok=True)
