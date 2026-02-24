import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--evaluate", "-e", action="store_true")
    parser.add_argument("--graph", "-g", action="store_true")
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--visual", "-v", action="store_true")
    parser.add_argument("--data-config", "-d", default=None)
    parser.add_argument("--model-config", "-m", default=None)
    parser.add_argument("--output-config", "-o", default=None)
    parser.add_argument("--probe-config", "-p", default=None)
    parser.add_argument("--directory-config", default=None)
    return parser