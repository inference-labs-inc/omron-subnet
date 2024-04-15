import argparse

import ezkl

if name == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--signature", type=str, required=True)
    parser.add_argument("--network", type=str, required=True)
    args = parser.parse_args()
