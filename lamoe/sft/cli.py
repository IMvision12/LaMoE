"""SFT data preparation and the existing local training entry point."""

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("prepare", help="Load and tokenize main or identity Hub data")
        sub.add_parser("train", help="Run local SFT with prepared files")
        parser.print_help()
        return 0
    if argv[0] == "prepare":
        from .hub import main as prepare_main

        return prepare_main(argv[1:])
    if argv[0] == "train":
        argv = argv[1:]
    from .train import main as train_main

    return train_main(argv)
