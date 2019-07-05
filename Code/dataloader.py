import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
import numpy as np


def get_char_set(train_tgt):
    tscp_str = [[x.decode("utf-8").lower() for x in train_tgt[i]] for i in range(len(train_tgt))]
    tscp_str = ["".join(x) if len(x) > 0 else x for x in tscp_str]
    symbol_set = set("".join(tscp_str))
    symbol_set.add("<blank>")
    symbol_set.add("<sos>")
    symbol_set.add("<eos>")
    symbol_set = sorted(list(symbol_set))
    symbol_dict = dict(zip(symbol_set, range(len(symbol_set))))
    symbol_dict["ignore"] = len(symbol_dict)
    return symbol_dict


class UtteranceDataset(Dataset):
    def __init__(self, symbol_dict, utter, tscp=None):
        super(UtteranceDataset, self).__init__()
        self.test = (tscp is None)
        self.utter = utter
        self.char_to_num_dict = symbol_dict
        self.num_to_char_dict = dict([(x[1], x[0]) for x in list(self.char_to_num_dict.items())])
        if not self.test:
            # to list of str
            tscp_output = []
            temp_tscp = [[x.decode("utf-8").lower() for x in tscp[i]] for i in
                         range(len(tscp))]
            # loop over all transcripts
            for s in temp_tscp:
                word_idx = [self.char_to_num_dict["<sos>"]]
                # loop over words
                for i in range(len(s)):
                    temp = [self.char_to_num_dict[char] for char in s[i]]
                    if i < len(s) - 1:
                        temp += [self.char_to_num_dict["<blank>"]]
                    word_idx += temp
                word_idx += [self.char_to_num_dict["<eos>"]]
                tscp_output.append(word_idx)
            self.tscp = tscp_output

    def __getitem__(self, i):
        if self.test:
            # return None for collation function convenience
            return self.utter[i], None
        return self.utter[i], self.tscp[i]

    def __len__(self):
        return len(self.utter)

    def get_dict(self):
        return self.char_to_num_dict, self.num_to_char_dict

    @classmethod
    def decode_symbols(cls, char_to_num_dict, num_to_char_dict, s):
        """

        :param char_to_num_dict: list of chars
        :param num_to_char_dict:
        :param s:
        :return:
        """
        if type(s) == torch.Tensor:
            s = s.detach().tolist()
        output = ""
        assert s[0] == char_to_num_dict["<sos>"]

        for i in range(1, len(s) - 1):
            if s[i] == char_to_num_dict["<blank>"]:
                output += " "
            elif s[i] == char_to_num_dict["<sos>"] or s[i] == char_to_num_dict["ignore"]:
                continue
            else:
                output += str(num_to_char_dict[s[i]])
        return output


def collate_utterance(seq_list, ignore_idx):
    inputs, targets = zip(*seq_list)
    test_flag = targets[0] is None
    input_lens = [len(seq) for seq in inputs]
    max_x = max(input_lens)
    # sorted by input lens
    sorted_idx = np.argsort(-np.array(input_lens))
    inputs = [torch.Tensor(inputs[i]) for i in sorted_idx]

    # test dataset if targets is None
    if not test_flag:
        target_lens = [len(x) for x in targets]
        max_y = max(target_lens)
        targets = [targets[i] for i in sorted_idx]
        target_lens = [target_lens[i] for i in sorted_idx]
        # pad with ignored index
        targets_padded = np.ones((len(targets), max_y)) * ignore_idx
        for i in range(len(targets)):
            targets_padded[i, :len(targets[i])] = targets[i]
        targets = torch.Tensor(targets_padded).long()

    # pad the longest utterance
    len_mod = max_x % 8
    if len_mod != 0:
        inputs[0] = nn.ZeroPad2d((0, 0, 0, 8 - len_mod))(inputs[0])

    # # pack only inputs
    packed = rnn.pack_sequence(inputs)
    padded_inputs, input_lens = rnn.pad_packed_sequence(packed)
    if not test_flag:
        return padded_inputs, targets, input_lens, target_lens
    else:
        return padded_inputs, input_lens

