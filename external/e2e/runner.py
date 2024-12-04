import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import symbolicregression
import sympy
import torch


def get_argparser():
    parser = argparse.ArgumentParser(description='End-to-End Symbolic Regression with Transformer baseline runner')
    parser.add_argument('--ckpt', required=True, help='ckpt file path')
    parser.add_argument('--train', help='training file path')
    parser.add_argument('--val', help='validation file path')
    parser.add_argument('--test', required=True, help='test file path')
    parser.add_argument('--out', required=True, help='output file name (dir path should be specified in config file)')
    return parser


def load_dataset(dataset_file_path, delimiter=' '):
    tabular_dataset = np.loadtxt(dataset_file_path, delimiter=delimiter)
    return tabular_dataset[:, :-1], tabular_dataset[:, -1]


def train(model, dataset_file_path):
    train_samples, train_targets = load_dataset(dataset_file_path)
    estimator = symbolicregression.model.SymbolicTransformerRegressor(
        model=model,
        max_input_points=10000,
        n_trees_to_refine=10,
        rescale=True
    )
    print('Fitting')
    estimator.fit(train_samples, train_targets)
    replace_ops = {'add': '+', 'mul': '*', 'sub': '-', 'pow': '**', 'inv': '1/', 'arctan':'atan'}
    eq_str = estimator.retrieve_tree(with_infos=True)['relabed_predicted_tree'].infix()
    for op, replace_op in replace_ops.items():
        eq_str = eq_str.replace(op, replace_op)
    return eq_str


def save_obj(obj, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def e2e_w_transformer2sympy(eq_str, threshold = 1e-2):
    eq_str_w_normalized_vars = re.sub(r'\bx_([0-9]*[0-9])\b', r'x\1', eq_str)
   # eq_str_w_normalized_vars = eq_str_w_normalized_vars.replace('arctan', 'atan')
    sympy_expr = sympy.parse_expr(eq_str_w_normalized_vars)
    updated_expr = sympy_expr
    while False:
        print("Simplifying once")
        print(sympy_expr)
        updated_expr = sympy_expr.xreplace({c: 0 for c in sympy_expr.atoms(sympy.Number) if abs(float(c)) < threshold})
        updated_expr = sympy.simplify(updated_expr)
        print(updated_expr)
        if sympy_expr == updated_expr: break
        sympy_expr = updated_expr

    return updated_expr


def main(args):
    print(args)
    ckpt_file_path = os.path.expanduser(args.ckpt)
    if torch.cuda.is_available():
        model = torch.load(ckpt_file_path, map_location=torch.device('cuda'))
        model = model.cuda()
    else:
        model = torch.load(ckpt_file_path, map_location=torch.device('cpu'))

    eq_str = train(model, args.train)
    sympy_eq = e2e_w_transformer2sympy(eq_str)
    print(sympy_eq)
    eq_file_path = args.out + '-est_eq.pkl'
    save_obj(sympy_eq, eq_file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
