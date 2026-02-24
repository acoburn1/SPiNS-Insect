from mimetypes import suffix_map
from sys import path_importer_cache
from tkinter import HIDDEN
from turtle import st
import torch
from torch import kl_div, nn
import time
import os
import argparse
from DriverUtils.Parser import get_parser
from DriverUtils.RunConfig import *


parser = get_parser()
args, _unknown = parser.parse_known_args()

runcfg = RunConfig(args)

if args.all:
    runcfg.train()
    runcfg.evaluate()
    runcfg.graph()
else:
    if args.train:
        runcfg.train()
    if args.evaluate:
        runcfg.evaluate()
    if args.graph:
        runcfg.graph()





