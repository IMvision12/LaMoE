"""Generate from local LaMoE weights and tokenizer. No identity facts are injected."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gdn-backend", choices=["reference", "fla"], default="reference"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use native BOS completion instead of user/assistant chat",
    )
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    from tokenizers import Tokenizer
    from .chat import assistant_prefix
    from .runtime import load_local, greedy_ids

    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.encode_special_tokens = True
    if args.raw:
        prefix = [tok.token_to_id("<|bos|>")] + tok.encode(
            args.prompt, add_special_tokens=False
        ).ids
    else:
        prefix = assistant_prefix(
            {"messages": [{"role": "user", "content": args.prompt}]}, tok
        )
    model = load_local(
        args.checkpoint, args.device, args.gdn_backend, args.expected_sha256
    )
    stops = {tok.token_to_id(t) for t in ["<|eos|>", "<|end_turn|>"]}
    if (
        None in stops
        or None in prefix
        or tok.get_vocab_size() != model.config.vocab_size
    ):
        raise ValueError("Tokenizer vocabulary/special tokens do not match this model")
    ids, reason = greedy_ids(model, prefix, args.max_new_tokens, stops)
    visible = ids[:-1] if ids and ids[-1] in stops else ids
    print(
        json.dumps(
            {
                "text": tok.decode(visible, skip_special_tokens=False),
                "generated_token_ids": ids,
                "prompt_tokens": len(prefix),
                "generated_tokens": len(ids),
                "stop_reason": reason,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
