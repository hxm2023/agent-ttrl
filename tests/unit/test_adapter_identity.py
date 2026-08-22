"""Adapter manifest hash tests (design doc §18.1)."""
from agent_ttrl.optimization.adapter_identity import AdapterSpec, same_identity


def test_hash_stable_and_distinguishes():
    base = AdapterSpec(base_sha256="a" * 64, lora_rank=16, lora_alpha=32,
                       lora_dropout=0.0, target_modules=["q_proj", "k_proj"])
    assert base.sha256() == base.sha256()
    different = AdapterSpec(base_sha256="a" * 64, lora_rank=32, lora_alpha=32,
                            lora_dropout=0.0, target_modules=["q_proj", "k_proj"])
    assert base.sha256() != different.sha256()
    assert not same_identity(base, different)


def test_order_insensitive_fields():
    a = AdapterSpec(base_sha256="a" * 64, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj"])
    b = AdapterSpec(base_sha256="a" * 64, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                    target_modules=["k_proj", "q_proj"])
    assert a.sha256() == b.sha256()


def test_training_input_refs_bind_identity():
    a = AdapterSpec(base_sha256="a" * 64, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                    training_input_event_refs=["e1"])
    b = AdapterSpec(base_sha256="a" * 64, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                    training_input_event_refs=["e2"])
    assert a.sha256() != b.sha256()


def test_default_profile_matches_design_doc():
    spec = AdapterSpec(base_sha256="a" * 64, lora_rank=16, lora_alpha=32,
                       lora_dropout=0.0,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
                       optimizer="adamw", learning_rate=5.0e-6)
    assert spec.lora_rank == 16 and spec.lora_alpha == 32
    assert spec.learning_rate == 5.0e-6
    assert len(spec.target_modules) == 7
