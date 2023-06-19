import logging
from typing import Dict, List, Optional, Tuple

import faiss
import faiss.contrib.torch_utils
import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq_easy_extend.models.transformer.decoder_continuous import ContinuousTransformerDecoder

logger = logging.getLogger("fairseq_easy_extend.models.transformer.optimised_decoder_continuous")
class OptimisedContinuousDecoder(ContinuousTransformerDecoder):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=False,
                 output_projection=None,
                 ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

        d = self.output_embed_dim
        self.k = 10

        # build a flat  index
        self.index_flat = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), d) if torch.cuda.is_available() else faiss.IndexFlatIP(d)
        self.index_flat.add(self.target_embed_normalized)


    def get_cosine_similarity_scores_optimised(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            features = F.normalize(query, p=2.0, dim=-1)
            distances, indices = self.index_flat.search(query.squeeze(dim=1), k=self.k)

            return indices, distances


    def get_cosine_similarity_scores(self, query: Tensor) -> Tensor:
        with torch.no_grad():
            features = F.normalize(query, p=2.0, dim=-1)

            distances, indices = self.index_flat.search(features.squeeze(dim=1), k=self.k)
            output = torch.zeros((features.shape[0], self.target_embed_normalized.shape[0]),
                                 device=self.target_embed_normalized.device, dtype=torch.float32)
            output.scatter_(1, indices, distances)

            return output.unsqueeze(1)

    def get_normalized_probs_optimised(
                self,
                net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
                log_probs: bool,
                sample: Optional[Dict[str, Tensor]] = None,
        ):
            with torch.no_grad():
                scores = self.get_cosine_similarity_scores_optimised(net_output[0])

            return scores