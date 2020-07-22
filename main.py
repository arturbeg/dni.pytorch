import argparse
from train import classifier
from train_semi_supervised import classifier as semi_supervised_classifier
import torch
from dataset import *
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DNI')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_unsupervised_epochs', type=int, default=50)
    parser.add_argument('--model_type', choices=['mlp', 'cnn'], default='mlp',
                    help='currently support mlp and cnn')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--conditioned', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--semi_supervised', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # do not support using mlp to trian cifar
    assert args.dataset != 'cifar10' or args.model_type != 'mlp'

    model_name = '%s.%s_dni'%(args.dataset, args.model_type, )
    if args.conditioned:
        model_name += '.conditioned'
    args.model_name = model_name
    if args.dataset == 'mnist':
        data = mnist(args)

    if args.semi_supervised:
        m = semi_supervised_classifier(args, data)
    else:
        m = classifier(args, data)

    m.train_model()
