""" model congfiguration"""

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class Gemma3TextConfig(Gemma3TextConfig):
    def __init__(self, train_adapters=False, **kwargs):
        super().__init__(**kwargs)
        self.train_adapters = train_adapters
