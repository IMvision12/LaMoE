"""Inspect architecture shapes without allocating the full parameter storage."""

import argparse
import json
from dataclasses import replace

import torch

from . import LaMoEConfig, LaMoEForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Inspect the small development configuration",
    )
    parser.add_argument(
        "--mla-head-gate-bias",
        type=float,
        help="Enable the optional MLA gate with an explicit initial bias",
    )
    args = parser.parse_args()
    config = LaMoEConfig.tiny() if args.tiny else LaMoEConfig()
    if args.mla_head_gate_bias is not None:
        config = replace(
            config, mla_head_gate=True, head_gate_bias_init=args.mla_head_gate_bias
        )
    with torch.device("meta"):
        model = LaMoEForCausalLM(config)
    print(json.dumps(model.parameter_summary(), indent=2))


if __name__ == "__main__":
    main()
