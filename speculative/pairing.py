"""F1: verify that the draft and target share a tokenizer (same ids for the same strings)."""


def compatibility(t_tok, d_tok, t_model, d_model) -> dict:
    tv, dv = t_tok.get_vocab(), d_tok.get_vocab()
    same_vocab = tv == dv
    same_special = t_tok.all_special_tokens == d_tok.all_special_tokens
    same_template = t_tok.chat_template == d_tok.chat_template
    t_emb, d_emb = t_model.config.vocab_size, d_model.config.vocab_size
    probe = "Patient denies smoking; HbA1c 7.2% on metformin 500 mg."
    same_ids = t_tok(probe)["input_ids"] == d_tok(probe)["input_ids"]
    return {
        "tokenizer_vocab_size": len(tv), "same_vocab": same_vocab, "same_special_tokens": same_special,
        "same_chat_template": same_template, "same_probe_ids": same_ids,
        "target_logit_dim": t_emb, "draft_logit_dim": d_emb,
        "compatible": same_vocab and same_special and same_ids and t_emb == d_emb,
    }
