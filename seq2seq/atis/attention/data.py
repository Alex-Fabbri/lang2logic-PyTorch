import time
import pickle as pkl
import torch
from util import SymbolsManager
from sys import path
import argparse
import random
import numpy as np

def process_train_data(opt):
    time_start = time.time()
    word_manager = SymbolsManager(True)
    word_manager.init_from_file("{}/vocab.q.txt".format(opt.data_dir), opt.min_freq, opt.max_vocab_size)
    form_manager = SymbolsManager(True)
    form_manager.init_from_file("{}/vocab.f.txt".format(opt.data_dir), 0, opt.max_vocab_size)
    print(word_manager.vocab_size)
    print(form_manager.vocab_size)
    data = []
    with open("{}/{}.txt".format(opt.data_dir, opt.train), "r") as f:
        for line in f:
            l_list = line.split("\t")
            w_list = word_manager.get_symbol_idx_for_list(l_list[0].strip().split(' '))
            r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
            data.append((w_list, r_list))
    out_mapfile = "{}/map.pkl".format(opt.data_dir)
    out_datafile = "{}/train.pkl".format(opt.data_dir)
    with open(out_mapfile, "wb") as out_map:
        pkl.dump([word_manager, form_manager], out_map)
    with open(out_datafile, "wb") as out_data:
        pkl.dump(data, out_data)

def serialize_data(opt):
    data = []
    managers = pkl.load( open("{}/map.pkl".format(opt.data_dir), "rb" ) )
    word_manager, form_manager = managers
    with open("{}/{}.txt".format(opt.data_dir, opt.test), "r") as f:
        for line in f:
            l_list = line.split("\t")
            w_list = word_manager.get_symbol_idx_for_list(l_list[0].strip().split(' '))
            r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
            data.append((w_list, r_list))
    out_datafile = "{}/test.pkl".format(opt.data_dir)
    with open(out_datafile, "wb") as out_data:
        pkl.dump(data, out_data)
    
   

main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument("-data_dir", type=str, default="../data/",
                                  help="data dir")
main_arg_parser.add_argument("-train", type=str, default="train",
                                  help="train dir")
main_arg_parser.add_argument("-test", type=str, default="test",
                                  help="test dir")
main_arg_parser.add_argument("-min_freq", type=int, default=2,
                                  help="minimum word frequency")
main_arg_parser.add_argument("-max_vocab_size", type=int, default=15000,
                                  help="max vocab size")
main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')

args = main_arg_parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
process_train_data(args)
serialize_data(args)
