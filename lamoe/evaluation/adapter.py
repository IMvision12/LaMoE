"""Register a local-only LaMoE model with lm-evaluation-harness.

The harness may separately acquire task data when invoked by its user. This
adapter never downloads model weights/tokenizers and never executes model output.
"""

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tokenizers import Tokenizer

from lamoe.runtime import load_local, greedy_ids
from lamoe.chat import assistant_prefix
from .scoring import encode_pair, continuation_score


@register_model("lamoe_local")
class LaMoELocal(LM):
    def __init__(
        self,
        pretrained,
        tokenizer,
        device="cpu",
        gdn_backend="reference",
        batch_size=1,
        max_gen_toks=256,
        chat=False,
        expected_sha256=None,
    ):
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("The portable adapter supports batch_size=1 only")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.encode_special_tokens = True
        self.model = load_local(pretrained, device, gdn_backend, expected_sha256)
        if self.tokenizer.get_vocab_size() != self.model.config.vocab_size:
            raise ValueError("Tokenizer/model vocabulary mismatch")
        if type(max_gen_toks) is not int or max_gen_toks <= 0:
            raise ValueError("max_gen_toks must be positive")
        self._max_gen_toks = max_gen_toks
        if isinstance(chat, str):
            if chat.lower() not in {"true", "false"}:
                raise ValueError("chat must be true or false")
            chat = chat.lower() == "true"
        self.chat = bool(chat)
        for token in ("<|bos|>", "<|eos|>", "<|end_turn|>"):
            if self.tokenizer.token_to_id(token) is None:
                raise ValueError("Frozen native chat special tokens are required")

    @property
    def eot_token_id(self):
        return self.tokenizer.token_to_id("<|eos|>")

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return next(self.model.parameters()).device

    def loglikelihood(self, requests, disable_tqdm=False):
        if self.chat:
            raise ValueError("Likelihood scoring uses raw prompts; choose chat=False")
        results = []
        for request in requests:
            context, continuation = request.args
            result = continuation_score(
                self.model, *encode_pair(self.tokenizer, context, continuation)
            )
            results.append(result)
            self.cache_hook.add_partial("loglikelihood", request.args, result)
        return results

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        stops = {self.eot_token_id, self.tokenizer.token_to_id("<|end_turn|>")}
        for request in requests:
            context, settings = request.args
            if settings.get("do_sample", False) or settings.get("temperature", 0) != 0:
                raise ValueError("Only deterministic greedy generation is implemented")
            if settings.get("repetition_penalty", 1.0) != 1.0:
                raise ValueError("Only repetition_penalty=1.0 is implemented")
            if self.chat:
                prefix = assistant_prefix(
                    {"messages": [dict(role="user", content=context)]}, self.tokenizer
                )
            else:
                prefix = [
                    self.tokenizer.token_to_id("<|bos|>")
                ] + self.tokenizer.encode(context, add_special_tokens=False).ids
            ids, _ = greedy_ids(
                self.model,
                prefix,
                settings.get("max_gen_toks", self.max_gen_toks),
                stops,
            )
            visible = ids[:-1] if ids and ids[-1] in stops else ids
            text = self.tokenizer.decode(visible, skip_special_tokens=False)
            until = settings.get("until", [])
            until = [until] if isinstance(until, str) else until
            offsets = [text.find(stop) for stop in until if stop and stop in text]
            if offsets:
                text = text[: min(offsets)]
            results.append(text)
            self.cache_hook.add_partial("generate_until", request.args, text)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError(
            "Rolling perplexity is not supported by the portable adapter"
        )
