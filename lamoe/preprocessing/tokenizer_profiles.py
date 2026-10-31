# The derived DeepSeek-V3 Split expressions below retain this upstream notice.
# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Local pre-tokenizer recipes; neither profile downloads a pretrained vocabulary."""

PROFILES = ("bytelevel", "deepseek_v3")
DEFAULT_PROFILE = "deepseek_v3"
DEEPSEEK_SOURCE = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/"
    "e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json"
)
# Exact Split expressions and order from the pinned DeepSeek-V3 tokenizer above.
DEEPSEEK_PATTERNS = (
    r"\p{N}{1,3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""".replace(
        r"\r", "\r"
    ).replace(r"\n", "\n"),
)


def make_pre_tokenizer(profile):
    from tokenizers import Regex, pre_tokenizers

    if profile == "bytelevel":
        return pre_tokenizers.ByteLevel(add_prefix_space=False)
    if profile == "deepseek_v3":
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(pattern), behavior="isolated")
                for pattern in DEEPSEEK_PATTERNS
            ]
            + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
        )
    raise ValueError(f"Unknown tokenizer profile: {profile}; choose from {PROFILES}")
