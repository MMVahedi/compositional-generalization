import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from icl_task_vectors import (
    PromptBuilder,
    TaskVectorConfig,
    TaskVectorExtractor,
    Injector,
    choose_backend,
)

# ----------------------------
# Prompt template helpers
# ----------------------------

def build_fewshot_prompt(pairs, query_x, sep, input_prefix="Input:", output_prefix="Output:"):
    """
    Each demo:
      Input: <x> <sep> Output: <y>
    Then query:
      Input: <query_x> <sep> Output:
    """
    chunks = []
    for x, y in pairs:
        chunks.append(f"{input_prefix} {x}\n{sep}\n{output_prefix} {y}\n")
    chunks.append(f"{input_prefix} {query_x}\n{sep}\n{output_prefix} ")
    return "\n".join(chunks)

def build_query_prompt(query_x, sep, input_prefix="Input:", output_prefix="Output:"):
    return f"{input_prefix} {query_x}\n{sep}\n{output_prefix} "


# ----------------------------
# Main demo
# ----------------------------

def main():
    # Change these to try other families:
    # - GPT-2 style: "gpt2"
    # - LLaMA: "meta-llama/Llama-2-7b-hf" (if you have access)
    # - Qwen: "Qwen/Qwen2-7B-Instruct" (example; use what you have)
    model_name = "gpt2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print(device)
    print(torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Some tokenizers (e.g. GPT-2) have no pad token; set it for generation convenience
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Separator: choose something that tokenizes reliably
    # (can be multi-token; our code finds all occurrences)
    sep = "->"

    # --- Few-shot demonstrations (toy task)
    # Task: map a country to its capital (simple associative mapping)
    demos = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
    ]

    # We'll extract a task vector from a prompt that contains multiple separator occurrences.
    # Then we will inject into a separate query prompt.
    extraction_query_x = "Spain"   # included just to complete the template

    fewshot_text = build_fewshot_prompt(demos, extraction_query_x, sep)
    query_text = build_query_prompt("Spain", sep)

    print("\n=== Few-shot prompt used for extraction ===\n")
    print(fewshot_text)

    print("\n=== Query prompt (no demos) ===\n")
    print(query_text)

    # ----------------------------
    # Encode prompts and find separator positions
    # ----------------------------
    pb = PromptBuilder(tokenizer, separator_text=sep)

    fewshot_enc = pb.encode(fewshot_text, device=device)
    query_enc = pb.encode(query_text, device=device)

    # We will inject at the separator position in the query prompt.
    # Usually there is exactly one separator in the query template.
    if len(query_enc.separator_positions) == 0:
        raise RuntimeError("No separator found in query prompt; check sep and template.")
    inject_pos = query_enc.separator_positions[-1]

    # ----------------------------
    # Choose which layer/block to use
    # ----------------------------
    # block_idx is 0..n_layers-1. Start mid-ish; you can sweep this later.
    # For GPT-2 (12 layers), try 6-10. For LLaMA/Qwen, start around 1/3 to 2/3 depth.
    block_idx = 10

    cfg = TaskVectorConfig(
        layer_idx=block_idx,              # transformer BLOCK index
        average_separators=False,         # default: last separator only
        # set True to average over all separators:
        # average_separators=True,
        normalize="l2",                   # optional; try None vs "l2"
        alpha=1.0,                        # injection strength; tune this
        device=device,
    )

    # ----------------------------
    # Extract task vector
    # ----------------------------
    extractor = TaskVectorExtractor(model, cfg)
    task_vec = extractor.extract(fewshot_enc, separator_text=sep)

    print("\n=== Extracted task vector meta ===")
    print(task_vec.meta)
    print("Vector norm:", float(task_vec.vector.norm().item()))

    # ----------------------------
    # Baseline generation (no injection)
    # ----------------------------
    base_out = model.generate(
        input_ids=query_enc.input_ids,
        attention_mask=query_enc.attention_mask,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)

    # ----------------------------
    # Injected generation
    # ----------------------------
    backend = choose_backend(model)  # auto-infer GPT2 vs LLaMA/Qwen structure
    injector = Injector(backend, cfg)

    injected_outputs = injector.inject_and_forward(
        model=model,
        prompt=query_enc,
        task_vector=task_vec,
        inject_position=inject_pos,
        use_cache=False,
    )

    # Continue generation from the injected forward pass by calling generate normally.
    # Simplest approach: just do generate with the hook active. We’ll do that instead:
    handle = backend.install_hook(
        model=model,
        layer_idx=task_vec.layer_idx,
        position=inject_pos,
        add_vector=task_vec.vector,
        alpha=cfg.alpha,
    )
    try:
        inj_out = model.generate(
            input_ids=query_enc.input_ids,
            attention_mask=query_enc.attention_mask,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        handle.remove()

    inj_text = tokenizer.decode(inj_out[0], skip_special_tokens=True)

    print("\n=== Baseline (no injection) ===\n")
    print(base_text)

    print("\n=== Injected ===\n")
    print(inj_text)

    print("\nTip: try changing block_idx, alpha, and average_separators.")


if __name__ == "__main__":
    main()

