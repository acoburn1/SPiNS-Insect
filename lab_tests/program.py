from DriverUtils.Parser import get_parser
from DriverUtils.RunConfig import *

parser = get_parser()
args, _ = parser.parse_known_args()

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
        runcfg.generate_output()





