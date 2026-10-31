"""Export a native local checkpoint for Transformers. No training, Hub or Git calls."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from transformers import GenerationConfig, PreTrainedTokenizerFast

from .configuration_lamoe import LaMoEConfig


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(
    checkpoint, tokenizer_file, output, expected_sha256=None, include_code=False
):
    checkpoint, tokenizer_file, output = (
        Path(checkpoint).resolve(),
        Path(tokenizer_file).resolve(),
        Path(output).resolve(),
    )
    if (
        output.is_relative_to(checkpoint)
        or checkpoint.is_relative_to(output)
        or tokenizer_file.is_relative_to(output)
    ):
        raise ValueError(
            "Export must be separate from original checkpoint/tokenizer files"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Use a new empty export directory; existing files are not overwritten"
        )
    weights = checkpoint / "model.safetensors"
    weight_sha = sha(weights)
    if expected_sha256 is not None and weight_sha != expected_sha256:
        raise ValueError("Checkpoint weight checksum mismatch")
    native = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    # Native configuration only: a re-export of an existing HF config must be
    # handled explicitly, rather than guessing how to strip its metadata.
    native["gdn_backend"] = "reference"
    config = LaMoEConfig(
        **native,
        architectures=["LaMoEForCausalLM"],
    )
    if include_code:
        config.auto_map = {
            "AutoConfig": "configuration_lamoe.LaMoEConfig",
            "AutoModelForCausalLM": "modeling_lamoe.LaMoEForCausalLM",
        }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=[
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end_turn|>",
        ],
        model_max_length=config.max_position_embeddings,
        padding_side="left",
        clean_up_tokenization_spaces=False,
    )
    native_tokens = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end_turn|>",
    ]
    if len(tokenizer) != config.vocab_size or any(
        tokenizer.convert_tokens_to_ids(token) != i
        for i, token in enumerate(native_tokens)
    ):
        raise ValueError(
            "Tokenizer vocabulary/native token IDs do not match this LaMoE checkpoint"
        )
    source_license = Path(__file__).resolve().parents[2] / "LICENSE"
    if source_license.is_file():
        license_text = source_license.read_text(encoding="utf-8")
    else:
        license_text = importlib.metadata.distribution("lamoe").read_text(
            "licenses/LICENSE"
        )
    if not license_text:
        raise ValueError("The source package license is missing; no export was created")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        shutil.disk_usage(output.parent).free
        < weights.stat().st_size + tokenizer_file.stat().st_size + 100_000_000
    ):
        raise ValueError("Insufficient free space for an independent export copy")
    output.mkdir(exist_ok=True)
    package = Path(__file__).resolve().parent
    if include_code:
        model_dir = package.parent / "model"
        for source in model_dir.glob("*.py"):
            if source.name in {"__init__.py", "__main__.py"}:
                continue
            name = {
                "config.py": "core_config.py",
                "modeling_lamoe.py": "modeling_lamoe_core.py",
            }.get(source.name, source.name)
            text = source.read_text(encoding="utf-8").replace(
                "from .config import", "from .core_config import"
            )
            (output / name).write_text(text, encoding="utf-8")
        replacements = {
            "from ..model.config import": "from .core_config import",
            "from ..model.cache import": "from .cache import",
            "from ..model.masking import": "from .masking import",
            "from ..model.modeling_lamoe import": "from .modeling_lamoe_core import",
        }
        for name in ["configuration_lamoe.py", "modeling_lamoe.py", "caching_lamoe.py"]:
            text = (package / name).read_text(encoding="utf-8")
            for old, new in replacements.items():
                text = text.replace(old, new)
            (output / name).write_text(text, encoding="utf-8")
        (output / "__init__.py").write_text('"""LaMoE Transformers model export."""\n')
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = (package / "chat_template.jinja").read_text(
        encoding="utf-8"
    )
    tokenizer.save_pretrained(output)
    # Keep original frozen tokenizer/weight bytes, including serialization.
    shutil.copyfile(tokenizer_file, output / "tokenizer.json")
    shutil.copyfile(weights, output / "model.safetensors")
    config.save_pretrained(output)
    GenerationConfig(
        do_sample=False,
        max_new_tokens=256,
        bos_token_id=2,
        eos_token_id=[3, 10],
        pad_token_id=0,
        use_cache=True,
    ).save_pretrained(output)
    (output / "requirements.txt").write_text(
        "transformers==5.17.0\ntorch>=2.9,<3\nsafetensors>=0.4.3,<1\n"
        "tokenizers>=0.23,<0.24\nnumpy>=1.26,<3\nhuggingface_hub>=0.34,<2\n"
    )
    (output / "LICENSE").write_text(license_text, encoding="utf-8")
    # A minimal model card, with no private training metadata or inferred scores.
    usage = (
        "Load with `AutoTokenizer.from_pretrained(path, trust_remote_code=True)` and\n"
        "`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.\n"
        if include_code
        else "Install the LaMoE source from https://github.com/IMvision12/LaMoE first.\n"
        "Then call `from lamoe.transformers import register; register()` before\n"
        "loading with `AutoTokenizer` and `AutoModelForCausalLM`, using\n"
        "`trust_remote_code=False`. This Hub layout contains no Python code.\n"
    )
    (output / "README.md").write_text(
        "---\nlibrary_name: transformers\npipeline_tag: text-generation\ntags:\n- custom-code\n---\n\n"
        "# LaMoE Transformers export\n\n"
        + usage
        + "Use `tokenizer.apply_chat_template` with `add_generation_prompt=True`.\n\n"
        "The original model weights and tokenizer are copied without numerical changes.\n"
        "Requires Transformers 5.17.0. CPU reference GDN is the default. CUDA/FLA\n"
        "requires separate validation. Padded batches use uncached generation.\n"
        "No datasets or collection/training code are included in this export.\n",
        encoding="utf-8",
    )
    if sha(output / "model.safetensors") != weight_sha or sha(
        output / "tokenizer.json"
    ) != sha(tokenizer_file):
        raise ValueError("Export copy failed integrity verification")
    manifest = dict(
        format="lamoe-transformers-export-v1"
        if include_code
        else "lamoe-model-files-v1",
        requires_lamoe_package=not include_code,
        weights_unchanged=True,
        tokenizer_unchanged=True,
        tested_transformers="5.17.0",
        self_excluded=True,
        files={
            p.relative_to(output).as_posix(): dict(
                bytes=p.stat().st_size, sha256=sha(p)
            )
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="Optional standalone remote-code export; default keeps all Python code in the source repository",
    )
    args = parser.parse_args(argv)
    result = export(
        args.checkpoint,
        args.tokenizer,
        args.output,
        args.expected_sha256,
        include_code=args.include_code,
    )
    print(
        f"Verified local export: {result}. No model loading, training or upload performed."
    )


if __name__ == "__main__":
    main()
