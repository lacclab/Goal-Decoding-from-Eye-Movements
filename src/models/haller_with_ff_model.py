# this model is the same as plm_as_f_model.py but replaces
# the fixations features with those from the fixations block of the beyelstm model

import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
)
from typing import List
import matplotlib.pyplot as plt
import string

from src.configs.trainer_args import Base
from src.models.base_model import BaseModel
from src.configs.constants import (
    MAX_SCANPATH_LENGTH,
    PredMode,
    NUM_ADDITIONAL_FIXATIONS_FEATURES,
)
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.PLMASFArgs import PLMASfArgs


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def pad_list(input_list, target_length, pad_with=0):
    # Calculate how many elements need to be added
    padding_length = target_length - len(input_list)

    # If padding_length is less than 0, the list is already longer than target_length
    if padding_length < 0:
        print("The list is already longer than the target length.")
        return input_list

    # Add padding_length number of zeros to the end of the list
    padded_list = input_list + [pad_with] * padding_length

    return padded_list


def get_word_length(word):
    if word in ["<s>", "</s>", "<pad>"]:
        return 0
    else:
        return len(word.translate(str.maketrans("", "", string.punctuation)))


def align_word_ids_with_input_ids(
    tokenizer: RobertaTokenizerFast,
    input_ids: torch.Tensor,
    decoded_to_txt_input_ids: list,
):
    word_ids_sn_lst = []
    retokenized_sn = tokenizer(
        decoded_to_txt_input_ids,
        return_tensors="pt",
    )
    for i in range(input_ids.shape[0]):
        word_ids_sn_lst.append(retokenized_sn.word_ids(i)[1:-1])

    word_ids_sn = torch.tensor(word_ids_sn_lst).to(input_ids.device)

    return word_ids_sn


def get_sn_word_lens(input_ids: torch.Tensor, decoded_to_txt_input_ids: list):
    def compute_p_lengths(p, target_length):
        return pad_list([get_word_length(word) for word in p], target_length)

    target_len = input_ids.shape[1]
    sn_word_len = torch.tensor(
        [
            compute_p_lengths(paragraph, target_len)
            for paragraph in decoded_to_txt_input_ids
        ]
    ).to(input_ids.device)

    return sn_word_len


def convert_positions_to_words_sp(
    scanpath: torch.Tensor,
    decoded_to_txt_input_ids: List[List[str]],
    roberta_tokenizer_prefix_space: RobertaTokenizerFast,
):
    sp_tokens_strs = []
    for i in range(scanpath.shape[0]):
        curr_sp_tokens = [roberta_tokenizer_prefix_space.cls_token] + [
            decoded_to_txt_input_ids[i][word_i + 1]  # +1 to skip the <s> token
            for word_i in scanpath[i].tolist()
            if word_i != -1
        ]
        curr_sp_tokens_str = " ".join(curr_sp_tokens)
        sp_tokens_strs.append(curr_sp_tokens_str.split())

    return sp_tokens_strs


def eyettention_legacy_code(scanpath, fixation_features):
    # sp_pos is batch_data.scanpath, when adding 2 to each element that is not -1, add a 0 column at the beginning and add 1 to the wholte tensor
    sp_pos = scanpath.clone()
    sp_pos[sp_pos != -1] += 1
    sp_pos = torch.cat(
        (torch.zeros(sp_pos.shape[0], 1).to(sp_pos.device), sp_pos), dim=1
    )
    sp_pos += 1
    sp_pos = sp_pos.int()

    # unused_sp_ordinal_pos = batch_data.fixation_features[:, :, 0].int() #! TODO why not used? delete?

    sp_fix_dur = fixation_features[
        ..., 1
    ]  #! The feature order is hard coded in model_args. Make sure it's correct
    sp_landing_pos = fixation_features[..., 2]

    # add a column of zeros to both sp_fix_dur and sp_landing_pos to account for the <s> token
    sp_fix_dur = torch.cat(
        (torch.zeros(sp_fix_dur.shape[0], 1).to(sp_fix_dur.device), sp_fix_dur),
        dim=1,
    )
    sp_landing_pos = torch.cat(
        (
            torch.zeros(sp_landing_pos.shape[0], 1).to(sp_landing_pos.device),
            sp_landing_pos,
        ),
        dim=1,
    )

    return sp_pos, sp_fix_dur, sp_landing_pos


def calc_sp_word_input_ids(
    input_ids: torch.Tensor,
    decoded_to_txt_input_ids: List[List[str]],
    backbone: str,
    scanpath: torch.Tensor,
):
    """This function calculates the word input ids for the scanpath

    Args:
        input_ids (torch.Tensor): The word sequence input ids.
                Tensor of (batch_size, max_text_length_in_tokens)
        decoded_to_txt_input_ids (list): The decoded input ids.
                (list of lists of strings)
        max_eye_len (int): The maximum scanpath length in the current batch (not global)
        backbone (str):  The backbone of the model (roberta base/large/RACE)
        scanpath (torch.Tensor): A scanpath tensor containing the word indices in the scanpath order
                Tensor of (batch_size, max_scanpath_length_in_words)
    """
    SP_word_ids, SP_input_ids = [], []
    roberta_tokenizer_prefix_space = RobertaTokenizerFast.from_pretrained(
        backbone, add_prefix_space=True
    )

    sp_tokens_strs = convert_positions_to_words_sp(
        scanpath=scanpath,
        decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        roberta_tokenizer_prefix_space=roberta_tokenizer_prefix_space,
    )

    tokenized_SPs = roberta_tokenizer_prefix_space.batch_encode_plus(
        sp_tokens_strs,
        add_special_tokens=False,
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        is_split_into_words=True,
    )
    for i in range(scanpath.shape[0]):
        encoded_sp = tokenized_SPs["input_ids"][i]
        word_ids_sp = tokenized_SPs.word_ids(i)  # -> Take the <sep> into account
        word_ids_sp = [val if val is not None else -1 for val in word_ids_sp]

        SP_word_ids.append(word_ids_sp)
        SP_input_ids.append(encoded_sp)

    word_ids_sp = torch.tensor(SP_word_ids).to(input_ids.device)
    sp_input_ids = torch.tensor(SP_input_ids).to(input_ids.device)

    return word_ids_sp, sp_input_ids


class HallerWithFFModel(BaseModel):
    def __init__(
        self,
        model_args: PLMASfArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)

        self.model_args = model_args
        self.backbone = model_args.backbone
        self.freeze_bert = model_args.freeze

        self.fast_tokenizer = RobertaTokenizerFast.from_pretrained(self.backbone)
        self.pad_token_id = self.fast_tokenizer.pad_token_id

        # ? self.preorder = False

        self.classifier_head = nn.Linear(
            self.model_args.model_params.lstm_hidden_size * 2, self.num_classes
        )  # *2 for bidirectional
        self.bert_dim = model_args.text_dim
        self.max_seq_len = model_args.max_seq_len
        self.max_eye_len = model_args.max_eye_len

        encoder_config = RobertaConfig.from_pretrained(self.backbone)
        encoder_config.output_hidden_states = True
        # initiate Bert with pre-trained weights
        print("keeping Bert with pre-trained weights")
        self.bert_encoder: RobertaModel = RobertaModel.from_pretrained(
            self.backbone, config=encoder_config
        )  # type: ignore

        # freeze the parameters in Bert model
        # TODO Replace for with with torch nograd and eval()?
        if self.freeze_bert:
            for param in self.bert_encoder.parameters():  # type: ignore
                param.requires_grad = False

        # create fse_lstm
        self.fse_lstm = nn.LSTM(
            input_size=self.bert_dim + NUM_ADDITIONAL_FIXATIONS_FEATURES,
            hidden_size=self.model_args.model_params.lstm_hidden_size,
            num_layers=self.model_args.model_params.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.model_args.model_params.lstm_dropout,
        )

        self.save_hyperparameters()

    def pool_subword_to_word(self, subword_emb, word_ids_sn, target, pool_method="sum"):
        # batching computing
        # Pool bert token (subword) to word level
        if target == "sn":
            max_len = self.max_seq_len  # CLS and SEP included
        elif target == "sp":
            max_len = (
                word_ids_sn.max().item() + 1
            )  # +1 for the <s> token at the beginning
        else:
            raise NotImplementedError

        merged_word_emb = torch.empty(subword_emb.shape[0], 0, self.bert_dim).to(
            subword_emb.device
        )
        for word_idx in range(max_len):
            word_mask = (
                (word_ids_sn == word_idx)
                .unsqueeze(2)
                .repeat(1, 1, self.bert_dim)
                .to(subword_emb.device)
            )
            # pooling method -> sum
            if pool_method == "sum":
                pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(
                    1
                )  # [batch, 1, 1024]
            elif pool_method == "mean":
                pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(
                    1
                )  # [batch, 1, 1024]
            else:
                raise NotImplementedError
            merged_word_emb = torch.cat([merged_word_emb, pooled_word_emb], dim=1)

        mask_word = torch.sum(merged_word_emb, 2).bool()
        return merged_word_emb, mask_word

    def fixation_sequence_encoder(
        self,
        sn_emd,
        sn_mask,
        word_ids_sn,
        sn_word_len,
        sp_emd,
        sp_pos,
        sp_fix_dur,
        sp_landing_pos,
        word_ids_sp,
        scanpath,
        x_fixations,
    ):
        """A LSTM based encoder for the fixation sequence (scanpath)
        Args:
            sp_emd (torch.Tensor): A tensor containing the text input_ids ordered according to the scanpath
            sp_pos (torch.Tensor): The word index of each fixation in the scanpath (the word the fixation is on)
            sp_fix_dur (torch.Tensor): The total fixation duration of each word in the scanpath (fixation)
            sp_landing_pos (torch.Tensor): The landing position of each word in the scanpath (fixation)
            sp_mask (torch.Tensor): The mask for the scanpath
            word_ids_sp (torch.Tensor): The word index of each input_id in the scanpath
            scanpath (torch.Tensor): The scanpath tensor

        Returns:
            _type_: _description_
        """

        # used for computing sp_merged_word_mask
        x = self.bert_encoder.embeddings.word_embeddings(sp_emd)
        x[sp_emd == self.pad_token_id] = 0
        # Pool bert subword to word level for English corpus
        _, sp_merged_word_mask = self.pool_subword_to_word(
            x, word_ids_sp, target="sp", pool_method="sum"
        )

        with torch.no_grad():
            outputs = self.bert_encoder(input_ids=sn_emd, attention_mask=sn_mask)
        #  Make the embedding of the <pad> token to be zeros
        outputs.last_hidden_state[sn_emd == self.pad_token_id] = 0
        # Pool bert subword to word level for english corpus
        merged_word_emb, sn_mask_word = self.pool_subword_to_word(
            outputs.last_hidden_state, word_ids_sn, target="sn", pool_method="sum"
        )
        batch_index = torch.arange(scanpath.shape[0]).unsqueeze(1).expand_as(scanpath)
        scanpath_add1 = scanpath.clone()
        scanpath_add1[scanpath != -1] += 1
        word_emb_sn = merged_word_emb[
            batch_index, scanpath_add1
        ]  # [batch, max_sp_length, emb_dim]
        x = word_emb_sn
        # concatenate the fixation features to the word embeddings
        x = torch.cat(
            (x, x_fixations), dim=2
        )  # [batch, max_sp_length, emb_dim + num_fixation_features]

        # pass through the LSTM layer
        sorted_lengths, indices = torch.sort(
            (scanpath != -1).sum(dim=1), descending=True
        )
        x = x[
            indices
        ]  # reorder sequences according to the descending order of the lengths

        # Pass the entire sequence through the LSTM layer
        packed_x = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=sorted_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=True,
        )

        packed_output, (hn, cn) = self.fse_lstm(packed_x)
        unpacked_output = nn.utils.rnn.unpack_sequence(packed_output)

        # unpacked output is a list of batch_size tensors,
        # each tensor is shaped [seq_len, no.directions*hidden_size]
        # create a tensor of shape [batch_size, no.directions*hidden_size]
        # by taking the mean of each element in the list
        lstm_mean_hidden = torch.stack([torch.mean(t, dim=0) for t in unpacked_output])

        # reorder the hidden states to the original order
        lstm_mean_hidden = lstm_mean_hidden[
            torch.argsort(indices)
        ]  # Tested. Reorders correctly

        return lstm_mean_hidden

    def forward(
        self,
        sn_emd,
        sn_mask,
        word_ids_sn,
        sn_word_len,
        sp_emd,  # (Batch, Maximum length of the scanpath in TOKENS + 1)
        sp_pos,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        sp_fix_dur,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        sp_landing_pos,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        word_ids_sp,  # (Batch, Maximum length of the scanpath in TOKENS + 1)
        scanpath,  # (Batch, Maximum length of the scanpath in WORDS)
        x_fixations,  # Fixations tensor (batch size, MAX_SCANPATH_LEN, 4). Padded with 0s
    ):
        assert (
            sn_emd[:, 0].sum().item() == 0
        )  # The CLS token is always present first (and 0 in roberta)

        fse_output = self.fixation_sequence_encoder(
            sn_emd=sn_emd,
            sn_mask=sn_mask,
            word_ids_sn=word_ids_sn,
            sn_word_len=sn_word_len,
            sp_emd=sp_emd,
            sp_pos=sp_pos,
            sp_fix_dur=sp_fix_dur,
            sp_landing_pos=sp_landing_pos,
            word_ids_sp=word_ids_sp,
            scanpath=scanpath,
            x_fixations=x_fixations,
        )  # [batch, step, dec_o_dim]

        pred = self.classifier_head(fse_output)

        return pred, fse_output

    def shared_step(
        # TODO update similar to base_roberta.py for ordered classification
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            batch (tuple): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_

        Notes:
            - in input ids: 0 is for CLS, 2 is for SEP, 1 is for PAD
        """
        # (paragraph_input_ids,
        # paragraph_input_masks,
        # input_ids,
        # input_masks,
        # labels,
        # eyes,
        # answer_mappings,
        # fixation_features,
        # fixation_pads,
        # scanpath, #? Scanpath is in IA_IDs format, need to turn it to input_ids format using inversions and input ids
        # scanpath_pads,
        # inversions,
        # inversions_pads,
        # grouped_inversions,
        # trial_level_features)
        batch_data = self.unpack_batch(batch)
        assert batch_data.input_ids is not None, "input_ids not in batch_dict"
        assert batch_data.input_masks is not None, "input_masks not in batch_dict"
        assert batch_data.scanpath is not None, "scanpath not in batch_dict"
        assert batch_data.fixation_features is not None, "eyes_tensor not in batch_dict"
        assert batch_data.scanpath_pads is not None, "scanpath_pads not in batch_dict"
        assert batch_data.eyes is not None, "eye not in batch_dict"

        shortest_scanpath_pad = batch_data.scanpath_pads.min()
        longest_batch_scanpath: int = int(MAX_SCANPATH_LENGTH - shortest_scanpath_pad)

        scanpath = batch_data.scanpath[..., :longest_batch_scanpath]
        fixation_features = batch_data.fixation_features[
            ..., :longest_batch_scanpath, :
        ]
        x_fixations = fixation_features[..., :4]

        # scanpath masks
        sp_masks = torch.ones_like(scanpath)
        sp_masks[scanpath == self.pad_token_id] = 0

        decoded_to_txt_input_ids = self.fast_tokenizer.batch_decode(
            batch_data.input_ids, return_tensors="pt"
        )

        word_ids_sn = align_word_ids_with_input_ids(
            tokenizer=self.fast_tokenizer,
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        )

        # in the decoded texts, space between <pad><pad>, <pad><s>, etc.
        decoded_to_txt_input_ids = list(
            map(
                lambda x: x.replace("<", " <").split(" ")[1:],
                decoded_to_txt_input_ids,
            )
        )

        sn_word_len = get_sn_word_lens(
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        )

        word_ids_sp, sp_input_ids = calc_sp_word_input_ids(
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
            backbone=self.backbone,
            scanpath=scanpath,
        )

        sp_pos, sp_fix_dur, sp_landing_pos = eyettention_legacy_code(
            scanpath=scanpath,
            fixation_features=fixation_features,
        )

        sn_embd = batch_data.input_ids
        sn_mask = batch_data.input_masks
        # if the second dimension of the scanpath is more than the maximum context length (of self.bert_encoder), cut it and notify
        bert_encoder_max_len = self.bert_encoder.config.max_position_embeddings
        if sp_input_ids.shape[1] > bert_encoder_max_len - 1:
            print(
                f"Text length is more than the maximum context length of the model ({bert_encoder_max_len}). Cutting from the BEGINNING of the text to max length."
            )
            sn_embd = sn_embd[:, : bert_encoder_max_len - 1]
            sn_mask = sn_mask[:, : bert_encoder_max_len - 1]

        logits, fse_output = self(
            sn_emd=sn_embd,
            sn_mask=sn_mask,
            word_ids_sn=word_ids_sn,
            sn_word_len=sn_word_len,
            sp_emd=sp_input_ids,
            sp_pos=sp_pos,
            sp_fix_dur=sp_fix_dur,
            sp_landing_pos=sp_landing_pos,
            word_ids_sp=word_ids_sp,
            scanpath=scanpath,
            x_fixations=x_fixations,
        )

        labels = batch_data.labels

        if self.prediction_mode == PredMode.CONDITION:
            ordered_labels = labels
            ordered_logits = logits
        else:
            raise NotImplementedError("Prediction mode not implemented")

        if self.class_weights is not None:
            loss = self.calculate_weighted_loss(
                logits=logits, labels=labels, ordered_labels=ordered_labels
            )
        else:
            loss = self.loss(logits.squeeze(), labels)

        if self.model_args.add_contrastive_loss:
            # TODO contrastive loss
            raise NotImplementedError("Contrastive loss not implemented yet")

        return ordered_labels, loss, ordered_logits.squeeze(), labels, logits.squeeze()

    def order_labels_logits(self, logits, labels, answer_mapping):
        # Get the sorted indices of answer_mapping along dimension 1
        sorted_indices = answer_mapping.argsort(dim=1)
        # Use these indices to rearrange each row in logits
        ordered_logits = torch.gather(logits, 1, sorted_indices)
        ordered_label = answer_mapping[range(answer_mapping.shape[0]), labels]

        return ordered_label, ordered_logits
