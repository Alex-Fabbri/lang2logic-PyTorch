import argparse
import copy
from main import *
import torch
from tree import Tree

def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def do_generate(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint):
    # initialize the rnn state to all zeros
    prev_c  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    prev_h  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()
    # reversed order
    enc_w_list.append(word_manager.get_symbol_idx('<S>'))
    enc_w_list.insert(0, word_manager.get_symbol_idx('<E>'))
    end = len(enc_w_list)
    enc_outputs = torch.zeros((1, end, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()
    for i in range(end-1, -1, -1):
        cur_input = torch.tensor(np.array(enc_w_list[i]), dtype=torch.long)
        if using_gpu:
            cur_input = cur_input.cuda()
        prev_c, prev_h = encoder(cur_input, prev_c, prev_h)
        enc_outputs[:, i, :] = prev_h
    # decode
    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent":0, "child_index":1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <=100:
        s = queue_decode[head-1]["s"]
        parent_h = s[1]
        t = queue_decode[head-1]["t"]
        if head == 1:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('(')], dtype=torch.long)
        if using_gpu:
            prev_word = prev_word.cuda()
        i_child = 1
        while True:
            curr_c, curr_h = decoder(prev_word, s[0], s[1], parent_h)
            prediction = attention_decoder(enc_outputs, curr_h)
            s = (curr_c, curr_h)
            _, _prev_word = prediction.max(1)
            prev_word = _prev_word
            if int(prev_word[0]) == form_manager.get_symbol_idx('<E>') or t.num_children >= checkpoint["opt"].dec_seq_length:
                break
            elif int(prev_word[0]) == form_manager.get_symbol_idx('<N>'):
                #print("we predicted N");exit()
                queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index":i_child, "t": Tree()})
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    # refine the root tree (TODO, what is this doing?)
    for i in range(len(queue_decode)-1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"]-1]["t"].children[cur["child_index"]-1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-temperature', type=int, default=1, help='temperature of sampling')
    main_arg_parser.add_argument('-sample', type=int, default=0, help='0 to use max at each timestep (-beam_size=1), 1 to sample at each timestep, 2 to beam search')
    main_arg_parser.add_argument('-beam_size', type=int, default=20, help='beam size')
    main_arg_parser.add_argument('-display', type=int, default=1, help='whether display on console')
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/', help='data path')
    main_arg_parser.add_argument('-input', type=str, default='test.t7', help='input data filename')
    main_arg_parser.add_argument('-output', type=str, default='output/seq2seq_output.txt', help='input data filename')
    main_arg_parser.add_argument('-model', type=str, default='checkpoint_dir/model_seq2seq', help='model checkpoint to use for sampling')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')

    # parse input params
    args = main_arg_parser.parse_args()
    # TODO, if the encoder was trained on a GPU do I need to call cuda
    using_gpu = False
    if args.gpuid > -1:
        using_gpu = True
    # load the model checkpoint
    checkpoint = torch.load(args.model)
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    attention_decoder = checkpoint["attention_decoder"]
    # put in eval mode for dropout
    encoder.eval()
    decoder.eval()
    attention_decoder.eval()
    # initialize the vocabulary manager to display text
    managers = pkl.load( open("{}/map.pkl".format(args.data_dir), "rb" ) )
    word_manager, form_manager = managers
    # load data
    data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
    reference_list = []
    candidate_list = []
    with open(args.output, "w") as output:
        # TODO change when running full -- this is to just reproduce the error
        #for i in range(30,50):
        #for i in range(278,280):
        for i in range(len(data)):
            print("example {}\n".format(i))
            x = data[i]
            reference = x[1]
            candidate = do_generate(encoder, decoder, attention_decoder, x[0], word_manager, form_manager, args, using_gpu, checkpoint)
            candidate = [int(c) for c in candidate]


            num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== "(")
            num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== ")")
            diff = num_left_paren - num_right_paren
            #print(diff)
            if diff > 0:
                for i in range(diff):
                    candidate.append(form_manager.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]

            ref_str = convert_to_string(reference, form_manager)
            cand_str = convert_to_string(candidate, form_manager)

            reference_list.append(reference)
            candidate_list.append(candidate)
            # print to console
            if args.display > 0:
                print("results: ")
                print(ref_str)
                print(cand_str)
                print(' ')
            output.write("{}\n".format(ref_str))
            output.write("{}\n".format(cand_str))

        val_acc = util.compute_tree_accuracy(candidate_list, reference_list, form_manager)
        print("ACCURACY = {}\n".format(val_acc))
        output.write("ACCURACY = {}\n".format(val_acc))
