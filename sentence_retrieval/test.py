import torch
from pandas import np
from torch import nn


class BERTFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 sep_token_id,
                 pad_token_id,
                 pretrained_model_dir,
                 batch_norm=False,
                 device="gpu",
                 pooling_method="cls"):
        super(BERTFeatureExtractor, self).__init__()

        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.device = device
        self.pretrained_model_dir = pretrained_model_dir
        self.pooling_method = pooling_method

        # Load pretrained model
        # TODO: this requires changes for different HuggingFaceModels
        if os.path.exists(self.pretrained_model_dir):
            self.pretrained_model = HuggingFaceModel.from_pretrained(self.pretrained_model_dir)
        else:
            self.pretrained_model = HuggingFaceModel.from_pretrained(
                'bert-base-multilingual-cased',
                output_hidden_states=False,
                output_attentions=False,
                use_bfloat16=True,
                num_labels=3,
                finetuning_task="xnli"
            )

            os.makedirs(self.pretrained_model_dir)

            self.pretrained_model.save_pretrained(self.pretrained_model_dir)

        # self.pretrained_model.training = True

        """
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        """

        assert num_layers >= 0, 'Invalid layer numbers'

        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(self.pretrained_model.config.hidden_size, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())


    def forward(self, s1_input, s2_input=None, max_seq_length=0):
        # analyzing the output of the output seems that "output" is a tuple with 2 elements, where:
        # - first element with shape [1, num_tokens, embed_dim], which I believe that corresponds to the embs representations of the tokens
        # - second element with shape [1, embed_dim], which I have no idea what it is ...
        # Based on the paper, the "[CLS]" tokens contains the sentence level representation, so that's what we want to use as sentence encoder.

        # construct segment ids and mask
        #max_len = max(input[1])

        # token_type_ids = torch.full((len(input[0]), len(input[0][1])), 0, dtype=torch.long)
        # input_mask = torch.full((len(input[0]), len(input[0][1])), 1, dtype=torch.long)
        # TODO: this requires changes for different HuggingFaceModels
        # lang_ids = torch.full((len(input[0]), len(input[0][1])), self.pretrained_model.config.lang2id["en"], dtype=torch.long)

        ###X = bert_representation[0].sum(1).squeeze(1)

        # bert_representation_only_cls_token = bert_representation
        # net_output = self.fcnet(bert_representation_only_cls_token)

        ###net_output = self.fcnet(X / input[1].view(-1, 1).expand_as(X))

        # return net_output

        ###return self.fcnet(self.pretrained_model(input[0])[0][:, 0, :])

        ##### sep_tokens_first_ocurrences = [[idx for idx in range(len(x)) if x[idx] == opt.sep_token_id][0] for x in input[0]]

        if s2_input is None:
            t_input, t_lengths = s1_input

            batch_size = len(t_input)

            sent_length = np.max(np.asarray(t_lengths))

            if (max_seq_length > 0) and (sent_length > max_seq_length):
                sent_length = max_seq_length

            padded_input = torch.full((batch_size, sent_length), self.pad_token_id, dtype=torch.long).to(self.device)

            my_token_type_ids = torch.full((batch_size, sent_length), 0, dtype=torch.long).to(self.device)

            my_attention_mask = torch.full((batch_size, sent_length), 0, dtype=torch.long).to(self.device)

            for i in range(batch_size):
                t_len = min(t_lengths[i], sent_length)

                padded_input[i][:t_len] = torch.tensor(t_input[i][:t_len], dtype=torch.long)

                if t_lengths[i] > sent_length:
                    # last token must be the [SEP] token
                    padded_input[i][-1] = torch.tensor(self.sep_token_id, dtype=torch.long)

                my_attention_mask[i][:t_len] = torch.ones(t_len)

        else:
            t_input, t_lengths = s1_input
            h_input, h_lengths = s2_input

            batch_size = len(t_input)
            sent_length = np.max(np.add(np.asarray(t_lengths), np.asarray(h_lengths))) - 1

            if (max_seq_length > 0) and (sent_length > max_seq_length):
                sent_length = max_seq_length

            padded_input = torch.full((batch_size, sent_length), self.pad_token_id, dtype=torch.long).to(self.device)

            my_token_type_ids = torch.full((batch_size, sent_length), 1, dtype=torch.long).to(self.device)

            # my_attention_mask = torch.full((len(input[0]), len(input[0][1])), 1, dtype=torch.long).to(self.device)
            my_attention_mask = torch.full((batch_size, sent_length), 0, dtype=torch.long).to(self.device)

            for i in range(batch_size):
                t_len = min(t_lengths[i], sent_length)
                h_len = h_lengths[i] - 1  # do not consider the CLS token in the beginning of H, which will be ignored
                h_len = max(min(h_len, sent_length - t_len), 0)

                padded_input[i][:t_len] = torch.tensor(t_input[i][:t_len], dtype=torch.long)
                #padded_input[i][t_lengths[i]] = torch.tensor(self.sep_token_id, dtype=torch.long)

                if h_len > 0:
                    # t_len < sent_length
                    padded_input[i][t_len:t_len + h_len] = torch.tensor(h_input[i][1:h_len+1], dtype=torch.long)

                if t_lengths[i] + h_lengths[i] - 1 > sent_length:
                    # last token must be the [SEP] token
                    padded_input[i][-1] = torch.tensor(self.sep_token_id, dtype=torch.long)

                my_token_type_ids[i][:t_len] = torch.zeros(t_len)

                my_attention_mask[i][:t_len + h_len] = torch.ones(t_len + h_len)

        if self.pooling_method == "cls":
            return self.fcnet(
                self.pretrained_model(
                    # TODO: this requires changes for different HuggingFaceModels
                    input_ids=padded_input,
                    token_type_ids=my_token_type_ids,
                    attention_mask=my_attention_mask,
                    # langs=lang_ids
                )[0][:, 0, :])

        elif (self.pooling_method == "avg") or (self.pooling_method == "avg_sqrt"):

            token_embeddings = self.pretrained_model(
                # TODO: this requires changes for different HuggingFaceModels
                input_ids=padded_input,
                token_type_ids=my_token_type_ids,
                attention_mask=my_attention_mask,
                # langs=lang_ids
            )[0]

            input_mask_expanded = my_attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            """
            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)
            """

            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            if self.pooling_method == "avg":
                return self.fcnet(sum_embeddings / sum_mask)
            else:
                return self.fcnet(sum_embeddings / torch.sqrt(sum_mask))

        else:
            raise Exception("Invalid pooling method: {}".format(self.pooling_method))