# OV Detection
from methods.detectors.ov_prompt.model import build as ov_prompt_build
# from methods.detectors.ov_prompt.solq import build as solq_build

def build_model(args):
    available_methods = [
         'ov_prompt',
         'solq'
         ]

    if args.method not in available_methods:
        raise ValueError(f'method [{args.method}] is not supported')

    elif args.method == 'ov_prompt':
        return ov_prompt_build(args)

