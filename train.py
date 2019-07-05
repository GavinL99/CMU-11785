import os

import Levenshtein
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchviz import make_dot, make_dot_from_trace

from LAS.model import *
# from model import *


class ModelManager:
    def __init__(self, model, train_loader, eval_loader, load_model=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_losses = []
        self.val_losses = []
        self.val_edit_dist = []
        self.epochs = 0
        self.n_vocab = len(VOCAB_DICT)
        self.print_interval = 10

        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.8)
        self.train_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        if load_model:
            model_path = "./Models/model_epoch_{}.pth".format(load_model["epoch_num"])
            if torch.cuda.is_available():
                state = torch.load(model_path)
            else:
                state = torch.load(model_path, map_location='cpu')
            model_state = state["model"]
            opt_state = state["optimizer"]
            self.epochs = state["epoch"]
            self.model.load_state_dict(model_state)
            self.model.to(DEVICE)
            self.scheduler.load_state_dict(state["scheduler"])
            self.optimizer.load_state_dict(opt_state)
            # move optimizer params to cuda
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        else:
            self.model.init_weights()
            self.model.to(DEVICE)

    def train(self):
        self.model.train()
        epoch_loss = 0.0
        self.scheduler.step()

        for batch_num, (inputs, targets, input_lens, target_lens) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            # (B, L, N_vocab), (B, L)
            decode_output, tgt_expand, attention_score = self.model(inputs, input_lens, targets, target_lens)
            decode_output = decode_output.view(-1, self.n_vocab)
            tgt_flatten = tgt_expand.flatten()
            loss = self.train_criterion(decode_output, tgt_flatten)
            #             perplexity = np.exp((loss.cpu().detach() / sum(target_lens)).numpy())

            perplexity = np.exp((loss.cpu().detach()).numpy())
            #             loss /= n_batch

            loss.backward()
            # clipping
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 5.0)

            self.optimizer.step()
            epoch_loss += loss.item()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del inputs
            del targets
            if batch_num % self.print_interval == 0:
                print("Batch: {} / Total: {} Loss: {} Perplexity: {}".format(batch_num, str(
                    len(self.train_loader)), loss, perplexity))
                if batch_num > 0:
                    ModelManager.save_attention_plot(attention_score[:, 0, :].cpu().detach().numpy(), self.epochs,
                                                     batch_num)

                    ModelManager.plot_grad_flow(self.model.named_parameters(), self.epochs, batch_num)

        epoch_loss = epoch_loss / (batch_num + 1)
        self.train_losses.append(epoch_loss)
        self.epochs += 1

    def evaluate(self, beam_search=False):
        # greedy search
        self.model.eval()
        model.to(DEVICE)
        assert self.eval_loader.batch_size == 1
        perplexity_total = 0.0
        edit_dist_total = 0.0

        if not beam_search:
            for batch_num, (inputs, targets, input_lens, target_lens)                     in enumerate(self.eval_loader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                decode_output, perplexity = self.model.inference(inputs, input_lens, False)
                translated_output = UtteranceDataset.decode_symbols(VOCAB_DICT,
                                                                    NUM_TO_CHAR,
                                                                    decode_output)
                translated_tgt = UtteranceDataset.decode_symbols(VOCAB_DICT,
                                                                 NUM_TO_CHAR,
                                                                 targets[0])
                edit_dist_total += Levenshtein.distance(translated_output,
                                                        translated_tgt)
                perplexity_total += perplexity

                if batch_num % 200 == 0:
                    print("Target Transcript: {}\nOutput: {}".format(
                        translated_tgt, translated_output))

            perplexity_total /= len(self.eval_loader)
            edit_dist_total /= len(self.eval_loader)
            print('[VAL]  Epoch [%d/%d]   Perplexity: %.4f  Edit_Dist: %.4f'
                  % (self.epochs, NUM_EPOCHS, perplexity_total, edit_dist_total))
            return edit_dist_total, perplexity_total
        else:
            raise NotImplementedError

    def predict(self, test_loader, beam_search=False):
        # greedy search
        print("Prediction!")
        self.model.eval()
        model.to(DEVICE)
        output = []

        for batch_num, (inputs, input_lens) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            if not beam_search:
                assert self.eval_loader.batch_size == 1
                decode_output, _ = self.model.inference(inputs, input_lens, False)
            else:
                decode_output = self.model.inference(inputs, input_lens, True)

            translated_output = UtteranceDataset.decode_symbols(VOCAB_DICT,
                                                                NUM_TO_CHAR,
                                                                decode_output)
            output.extend(translated_output)
            if batch_num % 2 == 0:
                print("Predicted Transcript: {}".format(translated_output))
        return output

    def save(self):
        state = {
            "epoch": self.epochs,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        torch.save(state, "./Models/model_epoch_{}.pth".format(self.epochs))

    def set_teacher_force_rate(self, new_p):
        self.model.decoder.teacher_force_p = new_p
        print("New Teacher Forcing Rate: {}"
              .format(self.model.decoder.teacher_force_p))

    def apply_train_policy(self):
        """
        Set teacher force rate, gumbel noise
        :return:
        """
        if self.epochs <= 5:
            self.set_teacher_force_rate(1.0 + 0.1)
        elif self.epochs <= 15:
            self.set_teacher_force_rate(0.9)
        else:
            self.set_teacher_force_rate(0.8)

        if self.epochs <= 15:
            self.model.decoder.gumbel = False
        else:
            self.model.decoder.gumbel = True

    @classmethod
    def save_attention_plot(cls, attention_weights, epoch, batch_num):
        fig = plt.figure()
        plt.imshow(attention_weights)
        fig.savefig("../Attention_Plots/epoch%d-%d.png" % (epoch, batch_num))
        plt.close()

    @classmethod
    def plot_grad_flow(cls, named_parameters, epoch, batch_num):
        # pass
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        fig = plt.figure(figsize=(8, 8))
        plt.xticks(rotation=30)
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        fig.savefig("../Gradient_Plots/epoch%d-%d.png" % (epoch, batch_num), bbox_inches='tight')
        plt.close()


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SZ = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 30
MAX_DECODE_LEN = 300
N_ENCODER = 256
N_DECODER = 2 * N_ENCODER
N_ATTENTION = 128
N_EMBED = 256
TEACHER_FORCE_P = 0.1
ATTENTION = True
#
# os.chdir("LAS")
train_np = np.load("../data/train.npy", encoding="bytes", allow_pickle=True)
train_tgt = np.load("../data/train_transcripts.npy", encoding="bytes", allow_pickle=True)
eval_np = np.load("../data/dev.npy", encoding="bytes", allow_pickle=True)
eval_tgt = np.load("../data/dev_transcripts.npy", encoding="bytes", allow_pickle=True)

data = UtteranceDataset(get_char_set(train_tgt), train_np, train_tgt)
VOCAB_DICT, NUM_TO_CHAR = data.get_dict()
data_eval = UtteranceDataset(VOCAB_DICT, eval_np, eval_tgt)

collate_f = lambda x: collate_utterance(x, VOCAB_DICT["ignore"])
train_loader = DataLoader(data, shuffle=True, batch_size=BATCH_SZ,
                          collate_fn=collate_f, drop_last=True)
eval_loader = DataLoader(data_eval, shuffle=False, batch_size=1,
                         collate_fn=collate_f, drop_last=True)

encoder_params = {
    "n_frame": 40,
    "n_hid": N_ENCODER,
    "n_attention": N_ATTENTION
}

if ATTENTION:
    attention_params = {
        "n_encoder": N_ENCODER * 2,
        "n_decoder": N_DECODER,
        "n_attention": N_ATTENTION,
        "DEVICE": DEVICE
    }
else:
    attention_params = None

decoder_params = {
    "n_vocab": len(VOCAB_DICT),
    "vocab": VOCAB_DICT,
    "n_hid": N_DECODER,
    "n_embed": N_EMBED,
    "attention_param": attention_params,
    "max_decode_len": MAX_DECODE_LEN,
    "eos_idx": VOCAB_DICT["<eos>"],
    "sos_idx": VOCAB_DICT["<sos>"],
    "DEVICE": DEVICE,
    "WEIGHT_TYING": True,
    "BEAM_SIZE": None,
    "GUMBLE": False
}


model = LAS(encoder_params, decoder_params)
manager = ModelManager(model, train_loader, eval_loader, {"epoch_num": 26})
manager.set_teacher_force_rate(0.9)

# best_dist = None
# for epoch in range(NUM_EPOCHS):
#     print("Start Epoch: {}".format(manager.epochs))
#     # manager.apply_train_policy()
#     manager.train()
#     entropy_loss, edit_dist = manager.evaluate(False)
#
#     if best_dist is None or edit_dist < best_dist:
#         best_dist = edit_dist
#         print(
#             "Saving model, predictions and generated output for epoch " + str(
#                 epoch) + " with Edit Dist: " + str(best_dist))
#         manager.save()


test_np = np.load("../data/test.npy", encoding="bytes", allow_pickle=True)
data_test = UtteranceDataset(VOCAB_DICT, test_np, None)
test_loader = DataLoader(data_test, shuffle=False, batch_size=1,
                         collate_fn=collate_f)
# {0: "'", 1: '+', 2: '-', 3: '.', 4: '<blank>', 5: '<eos>', 6: '<sos>',
# 7: '_', 8: 'a', 9: 'b', 10: 'c', 11: 'd', 12: 'e', 13: 'f', 14: 'g',
# 15: 'h', 16: 'i', 17: 'j', 18: 'k', 19: 'l', 20: 'm', 21: 'n', 22: 'o',
# 23: 'p', 24: 'q', 25: 'r', 26: 's', 27: 't', 28: 'u', 29: 'v', 30: 'w', 31: 'x', 32: 'y', 33: 'z', 34: 'ignore'}
# print(NUM_TO_CHAR)

manager.model.eval()
pred_tscp = manager.predict(test_loader, True)

# output result
import pandas as pd
out_df = pd.DataFrame()
out_df['Id'] = np.arange(0, len(pred_tscp))
out_df['Predicted'] = [x.upper() for x in pred_tscp]
print(out_df.head())
out_df.to_csv('submission_guanfu_2.csv', index=None)

