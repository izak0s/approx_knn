import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
from torch import Tensor

from fairseq_easy_extend.models import register_model
from fairseq_easy_extend.models.transformer.optimised_decoder_continuous import OptimisedContinuousDecoder
from fairseq_easy_extend.models.transformer.transformer_continuous import ContinuousTransformerConfig, \
    TransformerModelContinuous

logger = logging.getLogger("fairseq_easy_extend.models.transformer.transformer_continuous_approx")

@dataclass
class OptimisedContinuousTransformerConfig(ContinuousTransformerConfig):
   pass


@register_model("continuous_transformer_optimised", dataclass=OptimisedContinuousTransformerConfig)
class TransformerContinuousOptimised(TransformerModelContinuous):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.cfg = cfg


    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        logger.info("new decoder, good stuff!")

        decoder = OptimisedContinuousDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
        )
        return decoder


    @torch.no_grad()
    def get_normalized_probs_optimised(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        return self.decoder.get_normalized_probs_optimised(net_output, log_probs, sample)