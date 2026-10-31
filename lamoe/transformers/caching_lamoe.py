"""Transformers Cache adapter retaining MLA, GDN and convolution histories."""

from dataclasses import replace

import torch
from transformers.cache_utils import Cache

from ..model.cache import GDNCache, LayerCache, MLACache, ModelCache


class LaMoECache(Cache):
    def __init__(self, state: ModelCache | None = None, max_length=2048):
        super().__init__(layers=[])
        self.state = state
        self.max_length = max_length

    def get_seq_length(self, layer_idx=0):
        return 0 if self.state is None else self.state.length

    def get_max_length(self, layer_idx=0):
        return self.max_length

    def get_max_cache_shape(self, layer_idx=0):
        return self.max_length

    @property
    def is_compileable(self):
        return False

    @property
    def is_initialized(self):
        return self.state is not None

    @property
    def max_batch_size(self):
        return 0 if self.state is None else self.state.document_ids.shape[0]

    @property
    def max_cache_len(self):
        return self.max_length

    def _map(self, function):
        if self.state is None:
            return
        layers = []
        for layer in self.state.layers:
            mla = MLACache(function(layer.mla.latent), function(layer.mla.rotary_key))
            gdn = None
            if layer.gdn is not None:
                old = layer.gdn
                gdn = GDNCache(
                    function(old.state),
                    function(old.conv_q),
                    function(old.conv_k),
                    function(old.conv_v),
                )
            layers.append(LayerCache(mla, gdn))
        self.state = replace(
            self.state,
            layers=tuple(layers),
            document_ids=function(self.state.document_ids),
            last_segment_ids=function(self.state.last_segment_ids),
            last_position_ids=function(self.state.last_position_ids),
        )

    def reorder_cache(self, beam_idx: torch.LongTensor):
        self._map(lambda tensor: tensor.index_select(0, beam_idx.to(tensor.device)))

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats):
        self._map(lambda tensor: tensor.repeat_interleave(repeats, dim=0))

    def reset(self):
        self.state = None

    def crop(self, max_length):
        raise NotImplementedError(
            "LaMoE recurrent states cannot be cropped; assisted/speculative decoding is unsupported"
        )
