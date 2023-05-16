import argparse
from loguru import logger

def main(args):
    logger.info(args)
    if args.model_type == 'opt':
        from src.modules.opt import auto_compress
        auto_compress(args)
    else:
        raise NotImplementedError

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--target-model', type=str, default='.cache/models/answer_verification')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--dataset', type=str, default='answer_verification')
    parser.add_argument('--wbits', nargs='+', type=int, default=[2,3,4])
    parser.add_argument('--sparsities', nargs='+', type=float, default=[0.1, 0.33, 0.5, 0.67, 0.9, 0.95, 0.99])
    parser.add_argument('--tol', type=float, default=0.2)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)