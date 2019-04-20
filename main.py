import os
import json
import argparse
from preprocess import preprocess
from dataset import DatasetReader, Dataset
from CSPF import CSPF

# TODO: readme


if __name__ == '__main__':
    parser = argparse.ArgumentParser('The argument for the CSPF model')
    parser.add_argument('--mode', type=str, default='', help='support prepro/run/eval')
    # the below is for preprocessing
    parser.add_argument('--pre_config', type=str, default='pre-config.json',
                        help='the config file for the preprocess')
    # the below is for train
    # build dataset
    parser.add_argument('--city_info_path', type=str, default='data/Houston/info.json', help='the select city dataset')
    parser.add_argument('--neg_num', type=int, default=5,
                        help='the negative number of a positive sample for AUC computation')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='the split ratio of the test dataset')
    parser.add_argument('--trans_thre', type=float, default=2, help='the threshold of the negative intention')
    parser.add_argument('--train_path', type=str, default='data/Houston/train.data', help='the saved train datasets path')
    parser.add_argument('--dev_path', type=str, default='data/Houston/dev.data', help='the saved dev datasets path')
    # model
    parser.add_argument('--save_path', type=str, default='model/cspf.npz', help='the model save path')
    parser.add_argument('--dim', type=int, default=50, help='the dimension of the latent vectors')
    parser.add_argument('--rho_g', type=float, default=5.0, help='the ratio of the group loss')
    parser.add_argument('--rho_l', type=float, default=3.0, help='the ratio of the location loss')
    parser.add_argument('--gamma', type=float, default=0.01, help='the regularization parameter')
    parser.add_argument('--iter_num', type=int, default=60000, help='the iteration number')
    parser.add_argument('--save_step', type=int, default=2000, help='save model by this step')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    if args.mode == 'prepro':
        with open(args.pre_config, 'r', encoding='utf-8') as fp:
            config = json.load(fp)
        preprocess(config)

    elif args.mode == 'run':
        if os.path.exists(args.train_path) and os.path.exists(args.dev_path):
            train = Dataset(args.trans_thre).load(args.train_path)
            test = Dataset(args.trans_thre).load(args.dev_path)
        else:
            dataloader = DatasetReader(args.city_info_path, args.neg_num, args.test_ratio, args.trans_thre)
            train, test = dataloader.read()
            # save the split dataset
            train.save(args.train_path)
            test.save(args.dev_path)

        model = CSPF(dataset=train, save_path=args.save_path, k=args.dim, rho_g=args.rho_g, rho_l=args.rho_l,
                     gamma=args.gamma, iter_num=args.iter_num, save_step=args.save_step, lr=args.lr,
                     batch_size=args.batch_size)
        model.train({'train': train, 'eval': test}).save()
        # res = model.eval([test])
        # print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.mode == 'eval':
        test = Dataset(args.trans_thre).load(args.dev_path)

        model = CSPF(dataset=test, save_path=args.save_path, k=args.dim, rho_g=args.rho_g, rho_l=args.rho_l,
                     gamma=args.gamma, iter_num=args.iter_num, save_step=args.save_step, lr=args.lr,
                     batch_size=args.batch_size)
        model.load()
        res = model.eval([test])
        print(json.dumps(res, ensure_ascii=False, indent=2))
