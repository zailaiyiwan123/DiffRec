import contextlib
import datetime
import logging
import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
# from minigpt4.models.eva_vit import create_eva_vit_g
from transformers import BertTokenizer

import minigpt4.common.dist_utils as dist_utils
from minigpt4.common.dist_utils import download_cached_file
from minigpt4.common.logger import MetricLogger
from minigpt4.common.utils import is_url
from minigpt4.models.base_model import BaseModel
from minigpt4.models.rec_base_models import MatrixFactorization, LightGCN, SASRec, Personlized_Prompt, \
    random_mf, Soft_Prompt, RecEncoder_DIN, BiasEncoder, BiasSVD, BinMF


class Rec2Base(BaseModel):
    @classmethod
    def to_be_trained(self):
        pass

    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            # Compatible with PyTorch new API, avoid deprecation warnings
            try:
                return torch.amp.autocast(device_type="cuda", dtype=dtype)
            except Exception:
                return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    # @classmethod
    # def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    #     encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    #     encoder_config.encoder_width = vision_width
    #     # insert cross-attention layer every other block
    #     encoder_config.add_cross_attention = True
    #     encoder_config.cross_attention_freq = cross_attention_freq
    #     encoder_config.query_length = num_query_token
    #     Qformer = BertLMHeadModel(config=encoder_config)
    #     query_tokens = nn.Parameter(
    #         torch.zeros(1, num_query_token, encoder_config.hidden_size)
    #     )
    #     query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    #     return Qformer, query_tokens

    @classmethod
    def init_rec_encoder(self, rec_model, config, precision):
        if rec_model == "MF":
            print("### rec_encoder:", "MF")
            rec_model = MatrixFactorization(config)
        elif rec_model == "BinMF":
            print("### rec_encoder:", "BinMF")
            rec_model = BinMF(config)
        elif rec_model == "lightgcn":
            print("### rec_encoder:", "lightgcn")
            rec_model = LightGCN(config)
        elif rec_model == "sasrec":
            print("### rec_encoder:", "sasrec")
            rec_model = SASRec(config)
        elif rec_model == "DIN":
            print("### rec_encoder:", "DIN")
            rec_model = RecEncoder_DIN(config)
        elif rec_model == "personlized_prompt":
            print("### rec_encoder:", "personlized_prompt")
            rec_model = Personlized_Prompt(config)
        elif rec_model == "random_mf":
            print("### rec_encoder:", "random_mf")
            rec_model = random_mf(config)
        elif rec_model == 'soft_prompt':
            print("### rec_encoder:", "soft_prompt")
            rec_model = Soft_Prompt(config)
        elif rec_model == 'svd':
            print("### rec_encoder:", "bias_svd")
            rec_model = BiasSVD(config)
        else:
            rec_model = None
            warnings.warn(
                " the input rec_model is not MF, LightGCN or sasrec, or DCN, we won't utilize the rec_encoder directly.")
            # raise NotImplementedError("the current version olny supports the following models: MF,...")
        return rec_model

    @classmethod
    def init_bias_encoder(cls, config):
        return BiasEncoder(config)

    # @classmethod
    # def init_vision_encoder(
    #     cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    # ):
    #     assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
    #     visual_encoder = create_eva_vit_g(
    #         img_size, drop_path_rate, use_grad_checkpoint, precision
    #     )

    #     ln_vision = LayerNorm(visual_encoder.num_features)
    #     return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def after_evaluation(self, **kwargs):
        pass


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
