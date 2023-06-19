import json
import logging
from argparse import Namespace

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from fairseq_easy_extend.generators.OptimisedSequenceGenerator import OptimisedSequenceGenerator

logger = logging.getLogger("fairseq_easy_extend.tasks.OptimisedTranslationTask")


@register_task("optimised_translation", dataclass=TranslationConfig)
class OptimisedTranslationTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        logger.info("Init optimised translation task")

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        gen_args = json.loads(self.cfg.eval_bleu_args)
        self.seq_gen = self.build_generator(
            [model], Namespace(**gen_args), OptimisedSequenceGenerator
        )
        self.sequence_generator = self.seq_gen
        logger.info("Injected optimised sequence generator!")
        return model

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        result = super().inference_step(self.sequence_generator, models, sample, prefix_tokens, constraints)

        return result
