import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from LAS.dataloader import *

# from dataloader import *


class LAS(nn.Module):
    def __init__(self, encoder_param, decoder_param):
        super(LAS, self).__init__()
        self.encoder = Listener(encoder_param)
        self.decoder = Speller(decoder_param)

    def forward(self, inputs, input_lens, targets=None, target_lens=None):
        assert self.training
        encoder_output, input_condensed_lens, encoder_h, key, value = self.encoder(inputs, input_lens)
        # need both (encoder_output, input_condensed_lens) for attention
        # need to mask attention!
        return self.decoder(key, value, targets, input_condensed_lens, encoder_h)

    def inference(self, inputs, input_lens, beam_search=False):
        assert not self.training
        encoder_output, input_condensed_lens, encoder_h, key, value = self.encoder(inputs, input_lens)
        # need both (encoder_output, input_condensed_lens) for attention
        # need to mask attention!
        if not beam_search:
            return self.decoder.inference_greedy(key, value, input_condensed_lens, encoder_h)
        else:
            return self.decoder.inference_beam_search(key, value, input_condensed_lens, encoder_h, beam_size=64)

    def init_weights(self):
        print("Init Weights...")
        for m in self.modules():
            if type(m) == nn.LSTM or type(m) == nn.LSTMCell:
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
        output, (hn, cn) = self.lstm(packed, None)
        output, _ = pad_packed_sequence(output)
        return output, hn, new_lens


class Listener(nn.Module):
    def __init__(self, params):
        super(Listener, self).__init__()
        self.n_frame = params["n_frame"]
        self.n_hid = params["n_hid"]
        self.n_attention = params["n_attention"]
        # by default 4 layers
        self.lstm = nn.LSTM(self.n_frame, self.n_hid, 1, bidirectional=True,
                            dropout=0)
        # for bidirection
        self.pBLSTM1 = pBLSTM(2 * self.n_hid, self.n_hid)
        self.pBLSTM2 = pBLSTM(2 * self.n_hid, self.n_hid)
        self.pBLSTM3 = pBLSTM(2 * self.n_hid, self.n_hid)

        # for attention
        # (L, B, n_hid)
        key_linear_list = [nn.Linear(2 * self.n_hid, self.n_attention), nn.Tanh()]
        self.key_linear = nn.Sequential(*key_linear_list)
        value_linear_list = [nn.Linear(2 * self.n_hid, self.n_attention), nn.Tanh()]
        self.value_linear = nn.Sequential(*value_linear_list)

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
        key = self.key_linear(output)
        value = self.value_linear(output)
        return output, input_lens, hn, key, value


class Speller(nn.Module):
    def __init__(self, params):
        super(Speller, self).__init__()
        self.vocab = params["vocab"]
        self.n_vocab = params["n_vocab"]
        self.n_embed = params["n_embed"]
        self.n_hid = params["n_hid"]
        self.max_decode_len = params["max_decode_len"]
        self.eos_idx = params["eos_idx"]
        self.sos_idx = params["sos_idx"]
        self.DEVICE = params["DEVICE"]
        self.WEIGHT_TYING = params["WEIGHT_TYING"]
        self.beam_size = params["BEAM_SIZE"]
        self.gumble = params["GUMBLE"]
        
        # optional params:
        self.attention_param = params.get("attention_param")
        if self.attention_param is not None:
            print("Attention is On")
            self.n_attention = self.attention_param["n_attention"]
        else:
            print("Attention is Off")
        # self.embedding = nn.Embedding(self.n_vocab, self.n_embed)
        # use a linear layer instead of nn.Embedding for teacher forcing
        self.embedding = nn.Linear(self.n_vocab, self.n_embed)

        # context size + embedding size
        self.attention = Attention(self.attention_param)
        self.lstm1 = nn.LSTMCell(self.n_embed + self.n_attention, self.n_hid)
        self.lstm2 = nn.LSTMCell(self.n_hid, self.n_hid)

        query_linear_list = [nn.Linear(self.n_hid, self.n_attention), nn.Tanh()]
        self.query_linear = nn.Sequential(*query_linear_list)

        prob_linear_list = [nn.Dropout(0.1), nn.Linear(self.n_hid + self.n_attention, self.n_vocab)]
        self.prob_linear = nn.Sequential(*prob_linear_list)

        # self.teacher_linear = nn.Linear(self.n_hid + self.n_attention,
        #                                      self.n_embed)
        #
        # if self.WEIGHT_TYING:
        #     self.prob_distribution.weight = self.embedding.weight

    def forward(self, key, value, tscp_input, encoder_condensed_lens, encoder_h=None):
        """
        only for training
        :param key: attention key
        :param value: attention value
        :param tscp_input: padded target Tensor (B, max_y_len), padded by <eos>
        :param encoder_h
        :param encoder_condensed_lens: L // 8
        :return:
        """
        assert self.attention is not None

        n_batch, max_batch_len = tscp_input.shape
        attention_output = []
        target_pred = tscp_input[:, 1:]
        tscp_input = self.make_one_hot(tscp_input)
        tscp_input = self.embedding(tscp_input)
        # return for computing loss

        h1 = encoder_h.to(self.DEVICE).transpose(1, 0).reshape(n_batch, -1)
        c1 = torch.zeros_like(h1).to(self.DEVICE)
        # h1 = c1 = None
        h2 = c2 = None

        query = torch.zeros(n_batch, self.n_attention).to(self.DEVICE)
        # exclude last prediction of <eos>
        # (B, L_target-1, N_vocab)
        output_m = torch.zeros(n_batch, max_batch_len - 1, self.n_vocab).to(self.DEVICE)

        # (B, V) keep track of the previous logits
        logits = None

        # exclude the last <eos>
        for t in range(max_batch_len - 1):
            context, _ = self.attention(key, value, query, encoder_condensed_lens)
            # generate input to decoder for time t (B, E+A)

            # Teacher Forcing
            if t > 0 and self.teacher_force_p is not None and \
                    np.random.rand() > self.teacher_force_p:
                if self.gumble:
                    # (B, V) -> (B, E)
                    t_input = self.embedding(self.gumble_softmax(logits))
                else:
                    t_input = self.embedding(logits)
            else:
                t_input = tscp_input[:, t]

            # input with attention: (B, E + A)
            t_input = torch.cat((t_input, context), -1)
            # (B, n_hid)
            h1, c1 = self.lstm1(t_input, None if h1 is None else (h1, c1))
            h2, c2 = self.lstm2(h1, None if h2 is None else (h2, c2))

            query = self.query_linear(h2)
            context, attention_score = self.attention(key, value, query, encoder_condensed_lens)
            attention_output.append(attention_score)
            logits = self.prob_linear(torch.cat((h2, context), -1))
            output_m[:, t, :] = logits

        return output_m, target_pred

    def inference_greedy(self, key, value, encoder_condensed_lens, encoder_h):
        """

        :param key:
        :param value:
        :param encoder_condensed_lens:
        :param encoder_h:
        :return:
        """
        assert self.attention is not None
        n_batch = len(encoder_condensed_lens)
        assert n_batch == 1

        attention_output = []

        h1 = encoder_h.to(self.DEVICE).transpose(1, 0).reshape(n_batch, -1)
        c1 = torch.zeros_like(h1).to(self.DEVICE)
        # h1 = c1 = None
        h2 = c2 = None

        query = torch.zeros(n_batch, self.n_attention).to(self.DEVICE)
        # exclude last prediction of <eos>
        # (B, L_target-1, N_vocab)
        output_symbols = [self.vocab["<sos>"]]
        output_logits = []
        t_input = torch.Tensor([self.vocab["<sos>"]]).long().to(self.DEVICE)
        t = 1

        # exclude the last <eos>
        while t < self.max_decode_len:
            context, _ = self.attention(key, value, query, encoder_condensed_lens)
            # input with attention: (B, E + A)
            t_input = self.embedding(self.make_one_hot(t_input))
            t_input = torch.cat((t_input, context), -1)
            h1, c1 = self.lstm1(t_input, None if h1 is None else (h1, c1))
            # (B, n_hid)
            h2, c2 = self.lstm2(h1, None if h2 is None else (h2, c2))

            query = self.query_linear(h2)
            context, attention_score = self.attention(key, value, query, encoder_condensed_lens)
            attention_output.append(attention_score)

            logits = self.prob_linear(torch.cat((h2, context), -1))

            if self.gumble:
                # (B, V) -> (B, E)
                prob_output = self.gumble_softmax(logits)
            else:
                prob_output = F.softmax(logits, dim=1)

            prob, t_input = prob_output.max(dim=1)
            output_logits.append(prob.item())
            output_symbols.append(t_input.item())
            if t_input == self.vocab["<eos>"]:
                break
            t += 1

        if t == self.max_decode_len:
            output_symbols.append(self.vocab["<eos>"])
        # perplexity = np.exp(-F.log_softmax(torch.Tensor(output_logits), dim=-1).mean())

        return [output_symbols], attention_output

    def inference_beam_search(self, key, value, encoder_condensed_lens, encoder_h, beam_size=10):
        """
        :param key:
        :param value:
        :param encoder_condensed_lens:
        :param encoder_h:
        :return:
        """
        assert self.attention is not None
        n_batch = len(encoder_condensed_lens)
        assert n_batch == 1

        # bidirectional -> (B*k, v)
        h1 = encoder_h.to(self.DEVICE).squeeze().reshape(1, -1).repeat(beam_size, 1)
        # (L//8, B*K, A)
        key = key.repeat(1, beam_size, 1)
        value = value.repeat(1, beam_size, 1)
        c1 = torch.zeros_like(h1).to(self.DEVICE)
        # h1 = c1 = None
        h2 = c2 = None

        query = torch.zeros(beam_size, self.n_attention).to(self.DEVICE)

        # exclude last prediction of <eos>
        # (B * K, 1)
        output_symbols = [[self.vocab["<sos>"]] for _ in range(beam_size)]
        # (B, K)
        # log prob
        output_scores = torch.zeros(beam_size).to(self.DEVICE)
        # binary mask to decide whether end or not
        # if not hit <eos>, use 0
        output_end_flag = np.zeros(beam_size).astype(int)
        # (K) output sequence
        t_input = (torch.ones(beam_size) * self.vocab["<sos>"]).long().to(self.DEVICE)
        t = 0

        # exclude the last <eos>
        while t < self.max_decode_len:
            # need to update the batch length for attention
            context, _ = self.attention(key, value, query, encoder_condensed_lens * beam_size)
            # shape: (B*K, E)
            t_input = self.embedding(self.make_one_hot(t_input))
            # input with attention: (B*k, E + A)
            t_input = torch.cat((t_input, context), -1)
            h1, c1 = self.lstm1(t_input, None if h1 is None else (h1, c1))
            # (B*k, n_hid)
            h2, c2 = self.lstm2(h1, None if h2 is None else (h2, c2))

            query = self.query_linear(h2)
            context, _ = self.attention(key, value, query, encoder_condensed_lens * beam_size)

            # (B*K, V)
            logits = self.prob_linear(torch.cat((h2, context), -1))
            prob_output = F.softmax(logits, dim=1)
            # (B, K*V)
            prob_output = prob_output.view(-1)

            # mask for the first time to get different beams
            if t == 0:
                prob_output[self.n_vocab:] = 0.0

            # (K*V) repeat the same value n_vocab times
            cum_log_prob = output_scores.unsqueeze(1).repeat(1, self.n_vocab).reshape(-1)

            # consider early stopped seq (eos)
            for k in range(beam_size):
                # if already hit <eos>, not expand it
                # only keep the eos non zero
                if output_end_flag[k] == 1:
                    prob_output[k*self.n_vocab: (k+1)*self.n_vocab] = 0.0
                    prob_output[k*self.n_vocab + self.vocab["<eos>"]] = 1.0
                # shape (k*v, 1)
                # normalize by length but skip stopped ones
                prev_seq_len = len(output_symbols[k])
                normalize_mult = prev_seq_len / (prev_seq_len + 1.0) if \
                    output_end_flag[k] == 0 else 1.0
                cum_log_prob[k*self.n_vocab: (k+1)*self.n_vocab] *= normalize_mult
                cum_log_prob[k*self.n_vocab: (k+1)*self.n_vocab] += \
                    torch.log(prob_output[k*self.n_vocab: (k+1)*self.n_vocab] + 1e-30) / (prev_seq_len + 1.0)

            # update prob and select top K
            sorted_prob, sorted_idx = torch.sort(cum_log_prob, descending=True)
            sorted_prob = sorted_prob[:beam_size]
            sorted_idx = sorted_idx[:beam_size]
            # shape (K, 1)
            # range: 0 ~ K
            idx_to_expand = [x // self.n_vocab for x in sorted_idx]
            # range: 0 ~ n_vocab
            new_char = [x.item() % self.n_vocab for x in sorted_idx]
            end_flag = np.array([x == self.vocab["<eos>"] for x in new_char]).astype(int)

            # update for each sample
            # update score (K)
            # the beam score is ordered
            output_scores = sorted_prob
            # update flag
            output_end_flag = (output_end_flag[idx_to_expand] + end_flag > 0)
            # update input to LSTM
            h1 = h1[idx_to_expand]
            c1 = c1[idx_to_expand]
            h2 = h2[idx_to_expand]
            c2 = c2[idx_to_expand]
            query = query[idx_to_expand]
            # update sequence
            temp_new_seq = []
            for j in range(beam_size):
                temp_new_seq.append(output_symbols[idx_to_expand[j]] + [new_char[j]])
            output_symbols = temp_new_seq
            t_input = torch.Tensor(new_char).long().to(self.DEVICE)
            t += 1

            # check early early stop
            if output_end_flag.all():
                break
        # get the final output
        max_idx = torch.argmax(output_scores)
        # return [output_symbols[i*beam_size + max_idx[i]] for i in range(n_batch)]
        return [output_symbols[max_idx]]

    def gumble_softmax(self, logits, temperature=1, eps=1e-20):
        u_rand = torch.rand(logits.shape)
        gumble_noise = -torch.log(-torch.log(u_rand + eps) + eps)
        y = logits + gumble_noise.to(self.DEVICE)
        return F.softmax(y / temperature, dim=1)

    def make_one_hot(self, v):
        temp = torch.zeros(*v.shape, self.n_vocab)
        return temp.scatter_(v.dim(), v.unsqueeze(-1).cpu(), 1).to(self.DEVICE)


class Attention(nn.Module):
    def __init__(self, param):
        super(Attention, self).__init__()
        self.DEVICE = param["DEVICE"]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, key, value, query, encoder_condensed_lens):
        """

        :param key:
        :param value:
        :param query:
        :param encoder_condensed_lens:
        :return: output: (B, Attention_size)
        """
        # batch first
        n_batch = len(encoder_condensed_lens)
        # (B, U, A)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)
        # (B, 1, A)
        query = query.unsqueeze(1)
        # (B, U)
        energy = torch.bmm(key, query.transpose(1, 2)).squeeze(2)
        attention_wgt = self.softmax(energy)
        # masking
        mask = torch.zeros_like(attention_wgt).to(self.DEVICE)
        for i in range(n_batch):
            mask[i][:encoder_condensed_lens[i]] = 1
        attention_wgt = attention_wgt * mask
        attention_wgt = F.normalize(attention_wgt, p=1, dim=1)
        # (B, A)
        context = torch.bmm(attention_wgt.unsqueeze(1), value).squeeze(1)
        return context, attention_wgt
