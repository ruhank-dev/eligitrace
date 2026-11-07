"""Build a trainable PEFT model for a given precision strategy (shared by train + memory bench)."""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

STRATEGIES = ("fp32", "fp32-offload", "fp16-mixed", "bf16-mixed", "qlora-nf4")
AMP_DTYPE = {"fp16-mixed": torch.float16, "bf16-mixed": torch.bfloat16, "qlora-nf4": torch.bfloat16}


def lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_trainable(model_id: str, strategy: str, lora: dict, grad_ckpt: bool):
    """fp32       : base + LoRA in fp32 (reference precision; 1.5B base alone ~6.2 GB).
    fp32-offload: fp32 base split GPU/CPU by accelerate (layers streamed to GPU each forward).
    fp16-mixed : frozen base stored in fp16, LoRA params fp32, autocast fp16 + loss scaling.
    bf16-mixed : frozen base stored in bf16, LoRA params fp32, autocast bf16 (no loss scaling needed).
    qlora-nf4  : base quantised to 4-bit NF4 (bitsandbytes), bf16 compute, LoRA fp32."""
    if strategy == "qlora-nf4":
        from transformers import BitsAndBytesConfig

        q = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    elif strategy == "fp32-offload":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, device_map="auto",
            max_memory={0: "3GiB", "cpu": "14GiB"}, attn_implementation="sdpa")
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    else:
        dtype = {"fp32": torch.float32, "fp16-mixed": torch.float16, "bf16-mixed": torch.bfloat16}[strategy]
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
        model.to("cuda")
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, lora_config(lora))
    # keep trainable LoRA weights in fp32 regardless of base precision (stable optimiser state)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model
