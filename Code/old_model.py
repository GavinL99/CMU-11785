import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from LAS.dataloader import *
from LAS.dataloader import *

class LAS(nn.Module):
    def __init__(self, encoder_param, decoder_param):
        super(LAS, self).__init__()
        self.encoder = Listener(encoder_param)
        self.decoder = Speller(decoder_param)

    def forward(self, inputs, input_lens, targets=None, target_lens=None):
        assert self.training
        encoder_output, hn, _ = self.encoder(inputs, input_lens)
        return self.decoder(hn, targets, target_lens, encoder_output)

    def inference(self, inputs, input_lens, beam_search=False):
        _, hn, _ = self.encoder(inputs, input_lens)
        return self.decoder.inference(hn, beam_search)

    def init_weights(self):
        print("Init Weights...")
        for m in self.modules():
            if type(m) == nn.LSTM:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight.data)


class pBLSTM(nn.Module):
    def __init__(self, n_input, n_hid, dropout=0):
        super(pBLSTM, self).__init__()
        assert n_input == 2 * n_hid
        # for bidirection
        self.n_input = n_input
        self.n_hid = n_hid
        # * 2 for reshaping
        self.lstm = nn.LSTM(2 * self.n_input, self.n_hid, 1,
                            bidirectional=True, dropout=dropout)

    def forward(self, prev_output, prev_lens):
        '''
        :param prev_output: should call pad_packed_sequence outside
        :param prev_lens: will each time half the lens
        :return:
        '''
        # need to inverse pack_padded to transpose the tensor
        assert type(prev_output) == torch.Tensor
        n_batch = prev_output.size(1)
        new_input = prev_output.transpose(1, 0) \
            .reshape(n_batch, -1, 2 * self.n_input).transpose(1, 0)
        new_lens = [x // 2 for x in prev_lens]
        packed = pack_padded_sequence(new_input, new_lens)
        output, (hn, cn) = self.lstm(packed)
        output, _ = pad_packed_sequence(output)
        return output, hn, new_lens


class Listener(nn.Module):
    def __init__(self, params):
        super(Listener, self).__init__()
        self.n_frame = params["n_frame"]
        self.n_hid = params["n_hid"]
        # by default 4 layers
        self.lstm = nn.LSTM(self.n_frame, self.n_hid, 1, bidirectional=True,
                            dropout=0)
        # for bidirection
        self.pBLSTM1 = pBLSTM(2 * self.n_hid, self.n_hid)
        self.pBLSTM2 = pBLSTM(2 * self.n_hid, self.n_hid)
        self.pBLSTM3 = pBLSTM(2 * self.n_hid, self.n_hid)

    def forward(self, padded_inputs, input_lens):
        '''
        :param padded_inputs: already called pad_packed_sequence in collate f
        :param input_lens: 1/8 of initial lengths
        :return:
        '''
        input_lstm = pack_padded_sequence(padded_inputs, input_lens)
        output, _ = self.lstm(input_lstm)
        output, _ = pad_packed_sequence(output)
        output, _, input_lens = self.pBLSTM1(output, input_lens)
        output, _, input_lens = self.pBLSTM2(output, input_lens)
        output, hn, input_lens = self.pBLSTM3(output, input_lens)
        return output, hn, input_lens


class Speller(nn.Module):
    def __init__(self, params):
        super(Speller, self).__init__()
        self.n_vocab = params["n_vocab"]
        self.n_embed = params["n_embed"]
        self.n_hid = params["n_hid"]
        self.max_decode_len = params["max_decode_len"]
        self.eos_idx = params["eos_idx"]
        self.sos_idx = params["sos_idx"]
        self.DEVICE = params["DEVICE"]

        # optional params:
        self.attention_param = params.get("attention_param")
        if self.attention_param is not None:
            print("Attention is On")
            self.n_attention = self.attention_param["n_attention"]
        else:
            print("Attention is Off")
        self.teacher_force_p = params.get("teacher_force_p")

        self.embedding = nn.Embedding(self.n_vocab, self.n_embed)
        # context size + embedding size
        if self.attention_param is None:
            self.lstm1 = nn.LSTMCell(self.n_embed, self.n_hid)
            self.prob_distribution = nn.Linear(self.n_hid, self.n_vocab)
            self.attention = None
        else:
            self.attention = Attention(self.attention_param)
            self.lstm1 = nn.LSTMCell(self.n_embed + self.n_attention,
                                     self.n_hid)
            self.prob_distribution = nn.Linear(self.n_hid + self.n_attention,
                                               self.n_vocab)
        self.lstm2 = nn.LSTMCell(self.n_hid, self.n_hid)

    def _handle_variable_len_input(self, n_batch, encoder_h, tscp_input,
                                   tscp_lens):
        """
        Resort inputs by the length of Transcripts (instead of utterance)
        Expand targets to Max transcripts length and fill 0
        :param n_batch:
        :param encoder_h:
        :param tscp_input:
        :param tscp_lens:
        :return:
        """
        # sort input by transcript lengths first
        tscp_idx = np.argsort(-np.array(tscp_lens))
        # sorted
        tscp_lens = [tscp_lens[i] for i in tscp_idx]
        tscp_input = [tscp_input[i] for i in tscp_idx]
        encoder_h = encoder_h[tscp_idx, :]

        # exclude the last <eos>
        max_batch_len = tscp_lens[0] - 1
        tgt_expand = torch.zeros(n_batch * max_batch_len, requires_grad=False).long().to(self.DEVICE)
        for i in range(n_batch):
            start_i = i * max_batch_len
            # shift by 1 and exclude <sos>
            tgt_expand[start_i: start_i + tscp_lens[i] - 1] = tscp_input[i][1:]
        # apply embedding and maintain order
        tgt_expand = tgt_expand.clone()
        tscp_input = [self.embedding(x.long()) for x in tscp_input]
        return encoder_h, tscp_input, tscp_lens, tgt_expand, max_batch_len

    def forward(self, encoder_h, tscp_input, tscp_lens, encoder_output=None):
        # TODO: fix in-place ops
        """

        :param encoder_h: h output of encoder
        :param tscp_input: list of target transcripts
        :param tscp_lens: list of lens of targets
        :param encoder_output: for Attention
        :return: output_m (max_len-1 * n_batch * n_vocab),
        mask: vector (max_len-1 * n_batch)
        tgt_expand:
        """
        if self.attention is not None:
            assert encoder_output is not None
        n_batch = len(tscp_lens)
        num_active = n_batch - 1
        # reshape bidirection H
        encoder_h = encoder_h.transpose(1, 0).reshape(n_batch, self.n_hid)

        if self.attention is not None:
            attention_output = []
            context = torch.zeros(n_batch, self.n_attention).to(self.DEVICE)

        # resort by transcript length
        # max_batch_len = the true max size - 1 (need to make n-1 predictions)
        encoder_h, tscp_input, tscp_lens, tgt_expand, max_batch_len = \
            self._handle_variable_len_input(n_batch, encoder_h, tscp_input, tscp_lens)

        # init hidden values
        h1 = encoder_h.to(self.DEVICE)
        c1 = torch.zeros_like(h1).to(self.DEVICE)
        # h1 = c1 = None
        h2 = c2 = None
        output_m = torch.zeros(max_batch_len, n_batch, self.n_vocab, requires_grad=False).to(self.DEVICE)

        # exclude the last <eos>
        for t in range(max_batch_len - 1):
            # only use inputs that have valid lengths
            min_len_cap = tscp_lens[num_active]
            # in case there're inputs of same lengths
            while t >= min_len_cap - 1:
                num_active -= 1
                min_len_cap = tscp_lens[num_active]
            # generate input to decoder for time t (B, E or E+A)
            t_input = torch.zeros(n_batch, self.n_embed, requires_grad=False).to(self.DEVICE)
            # Teacher Forcing
            if t > 0 and self.teacher_force_p is not None and \
                    np.random.rand() < self.teacher_force_p:
                max_idx = output_m[t - 1, :, :].argmax(1)
                t_input[:num_active, :] = self.embedding(max_idx)[:num_active, :]
            else:
                # will have num_active seq inputs
                for i in range(num_active + 1):
                    t_input[i, :] = tscp_input[i][t, :]
            # input with attention: (B, E + A)
            t_input = t_input.clone().requires_grad_()
            if self.attention is not None:
                t_input = torch.cat((t_input, context), -1)

            # print("Context: ")
            # print(context)
#             print("t: {} h: ".format(t))
#             print(h1)

            h1, c1 = self.lstm1(t_input, None if h1 is None else (h1, c1))
            # (B, n_hid)
            h2, c2 = self.lstm2(h1, None if h2 is None else (h2, c2))
            if self.attention is not None:
                context, attention_score = self.attention(encoder_output, h2)
                attention_output.append(attention_score)
                output_m[t, :, :] = self.prob_distribution(
                    torch.cat((h2, context), -1))
            else:
                output_m[t, :, :] = self.prob_distribution(h2)

        # mask loss "vector" (need to reshape anyway, so just do it here)
        # (same for attention)
        mask = []
        for l in tscp_lens:
            mask += ([1] * (l - 1) + [0] * (max_batch_len - l + 1))

        mask = torch.Tensor(mask).float().requires_grad_().to(self.DEVICE)
        output_m = output_m.clone().requires_grad_()
        return output_m, mask, tgt_expand

    def inference(self, h, beam_search):
        encoder_h = h
        n_batch = encoder_h.size(1)
        assert n_batch == 1
        # reshape bidirection H
        encoder_h = encoder_h.transpose(1, 0).reshape(n_batch, self.n_hid)
        decode_output = [self.sos_idx]

        if not beam_search:
            # init hidden values
            h1 = encoder_h
            c1 = torch.zeros_like(h1)
            h2 = c2 = None
            decode_i = 0
            # input <sos>
            curr_char_idx = self.sos_idx

            while decode_i < self.max_decode_len and curr_char_idx != self.eos_idx:
                lstm_input = self.embedding(
                    torch.Tensor([curr_char_idx]).long())
                h1, c1 = self.lstm1(lstm_input, (h1, c1))
                h2, c2 = self.lstm2(h1, None if h2 is None else (h2, c2))
                logits = self.prob_distribution(h2)
                curr_char_idx = logits.squeeze(0).argmax().tolist()
                decode_output.append(curr_char_idx)
                decode_i += 1

            if curr_char_idx != self.eos_idx:
                decode_output.append(self.eos_idx)

        return decode_output




class Attention(nn.Module):
    def __init__(self, param):
        super(Attention, self).__init__()
        n_encoder_output = param["n_encoder"]
        n_decoder_output = param["n_decoder"]
        n_attention = param["n_attention"]
        self.key_linear = nn.Linear(n_encoder_output, n_attention)
        self.value_linear = nn.Linear(n_encoder_output, n_attention)
        self.query_linear = nn.Linear(n_decoder_output, n_attention)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_output, decoder_output):
        # batch first
        # (B, U, A)
        key = self.key_linear(encoder_output).transpose(1, 0)
        value = self.value_linear(encoder_output).transpose(1, 0)
        # (B, 1, A)
        query = self.query_linear(decoder_output).unsqueeze(1)
        # (B, U)
        energy = torch.bmm(key, query.transpose(1, 2)).squeeze(2)
        attention_wgt = self.softmax(energy)
        # (B, A)
        context = torch.bmm(attention_wgt.unsqueeze(1), value).squeeze(1)
        return context, attention_wgt
