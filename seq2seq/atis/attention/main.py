import argparse
import time
import pickle as pkl
import util
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim
import random

class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4*opt.rnn_size)
        if opt.dropoutrec > 0:
            self.dropout = nn.Dropout(opt.droputrec)

    def forward(self, x, prev_c, prev_h):
        gates = self.i2h(x) \
            + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        if self.opt.dropoutrec > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)  # n_b x hidden_dim
        return cy, hy

class RNN(nn.Module):
    def __init__(self, opt, input_size):
        super(RNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.lstm = LSTM(self.opt)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input_src, prev_c, prev_h):
        src_emb = self.embedding(input_src) # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h)
        return prev_cy, prev_hy

class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size

        self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top):
        # (batch*length*hidden) * (batch * hidden * 1) = (batch*length*1)
        #print("enc_s_top: {}\n".format(enc_s_top.size()))
        #print("dec_s_top: {}\n".format(dec_s_top.size()))
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        #dot = torch.legacy.nn.MM()((enc_s_top, torch.legacy.nn.View(opt.rnn_size,1).setNumInputDims(0)(dec_s_top)))
        #print("dot size: {}\n".format(dot.size()))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        #print("attention size: {}\n".format(attention.size()))

        #(batch*length*H)^T * (batch*length*1) = (batch*H*1)
        enc_attention = torch.bmm(enc_s_top.permute(0,2,1), attention)
        hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2),dec_s_top), 1)))
        h2y_in = hid
        if self.opt.dropout > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)
        return pred

def eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, form_manager):
    # encode, decode, backward, return loss
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()
    enc_batch, enc_len_batch, dec_batch = train_loader.random_batch()
    # do not predict after <E>
    enc_max_len = enc_batch.size(1)
    # because you need to compare with the next token!!
    dec_max_len = dec_batch.size(1) -1

    enc_outputs = torch.zeros((enc_batch.size(0), enc_max_len, encoder.hidden_size), requires_grad=True)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    enc_s = {}
    for j in range(opt.enc_seq_length + 1):
        enc_s[j] = {}

    dec_s = {}
    for j in range(opt.dec_seq_length + 1):
        dec_s[j] = {}

    for i in range(1, 3):
        enc_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
        dec_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
        if using_gpu:
            enc_s[0][i] = enc_s[0][i].cuda()
            dec_s[0][i] = dec_s[0][i].cuda()

    for i in range(enc_max_len):
        enc_s[i+1][1], enc_s[i+1][2] = encoder(enc_batch[:,i], enc_s[i][1], enc_s[i][2])
        enc_outputs[:, i, :] = enc_s[i+1][2]

    loss = 0

    for i in range(opt.batch_size):
        dec_s[0][1][i, :] = enc_s[enc_len_batch[i]][1][i, :]
        dec_s[0][2][i, :] = enc_s[enc_len_batch[i]][2][i, :]

    for i in range(dec_max_len):
        dec_s[i+1][1], dec_s[i+1][2] = decoder(dec_batch[:,i], dec_s[i][1], dec_s[i][2])
        pred = attention_decoder(enc_outputs, dec_s[i+1][2])
        loss += criterion(pred, dec_batch[:,i+1])

    loss = loss / opt.batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(decoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(),opt.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()
    return loss


def main(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    managers = pkl.load( open("{}/map.pkl".format(opt.data_dir), "rb" ) )
    word_manager, form_manager = managers
    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True
        torch.cuda.manual_seed(opt.seed)
    encoder = RNN(opt, word_manager.vocab_size)
    decoder = RNN(opt, form_manager.vocab_size)
    attention_decoder = AttnUnit(opt, form_manager.vocab_size)
    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention_decoder = attention_decoder.cuda()
    # init parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in attention_decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)

    #model_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    #params_encoder = sum([np.prod(p.size()) for p in model_parameters])
    #model_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
    #params_decoder = sum([np.prod(p.size()) for p in model_parameters])
    #model_parameters = filter(lambda p: p.requires_grad, attention_decoder.parameters())
    #params_attention_decoder = sum([np.prod(p.size()) for p in model_parameters])
    #print(params_encoder + params_decoder+ params_attention_decoder);exit()
    # 439254 as in D&L

    ##-- load data
    train_loader = util.MinibatchLoader(opt, 'train', using_gpu)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    ##-- start training
    step = 0
    epoch = 0
    optim_state = {"learningRate" : opt.learning_rate, "alpha" :  opt.decay_rate}
    # default to rmsprop
    if opt.opt_method == 0:
        print("using RMSprop")
        encoder_optimizer = optim.RMSprop(encoder.parameters(),  lr=optim_state["learningRate"], alpha=optim_state["alpha"])
        decoder_optimizer = optim.RMSprop(decoder.parameters(),  lr=optim_state["learningRate"], alpha=optim_state["alpha"])
        attention_decoder_optimizer = optim.RMSprop(attention_decoder.parameters(),  lr=optim_state["learningRate"], alpha=optim_state["alpha"])
    criterion = nn.NLLLoss(size_average=False, ignore_index=0)

    print("Starting training.")
    encoder.train()
    decoder.train()
    attention_decoder.train()
    iterations = opt.max_epochs * train_loader.num_batch
    for i in range(iterations):
        epoch = i // train_loader.num_batch
        start_time = time.time()
        #print("iteration: {}\n".format(i))
        train_loss = eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, form_manager)
        #exponential learning rate decay
        if opt.opt_method == 0:
            if i % train_loader.num_batch == 0 and opt.learning_rate_decay < 1:
                if epoch >= opt.learning_rate_decay_after:
                    decay_factor = opt.learning_rate_decay
                    optim_state["learningRate"] = optim_state["learningRate"] * decay_factor #decay it
                    for param_group in encoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in attention_decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]

        end_time = time.time()
        if i % opt.print_every == 0:
            print("{}/{}, train_loss = {}, time/batch = {}".format( i, iterations, train_loss, (end_time - start_time)/60))

        #on last iteration
        if i == iterations -1:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["attention_decoder"] = attention_decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/model_seq2seq_attention".format(opt.checkpoint_dir))

        if train_loss != train_loss:
            print('loss is NaN.  This usually indicates a bug.')
            break

if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/', help='data path')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')
    main_arg_parser.add_argument('-checkpoint_dir',type=str, default= 'checkpoint_dir', help='output directory where checkpoints get written')
    main_arg_parser.add_argument('-savefile',type=str, default='save',help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    main_arg_parser.add_argument('-print_every',type=int, default=2000,help='how many steps/minibatches between printing out the loss')
    main_arg_parser.add_argument('-rnn_size', type=int,default=200, help='size of LSTM internal state')
    main_arg_parser.add_argument('-num_layers', type=int, default=1, help='number of layers in the LSTM')
    main_arg_parser.add_argument('-dropout',type=float, default=0.4,help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    main_arg_parser.add_argument('-dropoutrec',type=int,default=0,help='dropout for regularization, used after each c_i. 0 = no dropout')
    main_arg_parser.add_argument('-enc_seq_length',type=int, default=50,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-dec_seq_length',type=int, default=100,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-batch_size',type=int, default=20,help='number of sequences to train on in parallel')
    main_arg_parser.add_argument('-max_epochs',type=int, default=80,help='number of full passes through the training data')
    main_arg_parser.add_argument('-opt_method', type=int,default=0,help='optimization method: 0-rmsprop 1-sgd')
    main_arg_parser.add_argument('-learning_rate',type=float, default=0.01,help='learning rate')
    main_arg_parser.add_argument('-init_weight',type=float, default=0.08,help='initailization weight')
    main_arg_parser.add_argument('-learning_rate_decay',type=float, default=0.98,help='learning rate decay')
    main_arg_parser.add_argument('-learning_rate_decay_after',type=int, default=5,help='in number of epochs, when to start decaying the learning rate')
    main_arg_parser.add_argument('-restart',type=int, default=-1,help='in number of epochs, when to restart the optimization')
    main_arg_parser.add_argument('-decay_rate',type=float, default=0.95,help='decay rate for rmsprop')
    main_arg_parser.add_argument('-grad_clip',type=int, default=5,help='clip gradients at this value')

    args = main_arg_parser.parse_args()
    main(args)
    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))
