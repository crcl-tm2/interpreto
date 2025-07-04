# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
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
"""
Definition of different granularity levels for explainers (tokens, words, sentences...)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

# Lazy spacy import for SENTENCE granularities
try:
    import spacy

    _HAS_SPACY = True
except ModuleNotFoundError:
    _HAS_SPACY = False


class NoWordIdsError(AttributeError):
    """
    Exception raised when the word_ids method is not available on the tokenizer
    """

    def __init__(self):
        super().__init__(
            "Word-level granularity level requires tokenization with a fast tokenizer (i.e., tokenizer.is_fast=True), "
            "because it relies on `.word_ids()` to associate tokens with words. "
            "Please either use a fast tokenizer or switch to token-level granularity."
        )


class GranularityMethodAggregation(Enum):
    """
    Enumeration of the available aggregation strategies for combining token-level
    scores into a single score for each unit of a higher-level granularity
    (e.g., word, sentence).

    This is used in explainability methods to reduce token-based attributions
    according to a defined granularity.

    Attributes:
        MEAN: Average of the token scores within each group.
        MAX: Maximum token score within each group.
        MIN: Minimum token score within each group.
        SUM: Sum of all token scores within each group.
        SIGNED_MAX_ABS: Selects the token with the highest absolute score and returns its signed value.
                        For example, given scores [3, -1, 7], returns 7; for [3, -1, -7], returns -7.
    """

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    SIGNED_MAX_ABS = "signed_max_abs"


class Granularity(Enum):
    """
    Enumerations of the different granularity levels supported for masking perturbations
    Allows to define token-wise masking, word-wise masking...
    """

    ALL_TOKENS = "all_tokens"  # All tokens, including special tokens like padding, eos, cls, etc.
    TOKEN = "token"  # Strictly tokens of the input
    WORD = "word"  # Words of the input
    SENTENCE = "sentence"  # Sentences of the input
    # PARAGRAPH = "paragraph"  # Not supported yet, the "\n\n" characters are replaced by spaces in many tokenizers.
    DEFAULT = ALL_TOKENS

    @staticmethod
    # @jaxtyped(typechecker=beartype)
    def get_indices(
        inputs: BatchEncoding,
        granularity: Granularity | None,
        tokenizer: PreTrainedTokenizer | None,
    ) -> list[list[list[int]]]:
        """
        Return *indices* of the tokens that correspond to the desired
        granularity for each samples.

        The result is a *list[list[list[int]]]* where each inner list contains the
        positions of the tokens that compose one granularity unit.
        The list hierarchy is as follows:

            - For each sample.

            - For each element for the granularity level. Thus, tokens, words, or sentences.

            - The inner list contains the positions of the tokens that compose one granularity unit.

        The granularity levels are:

            - ``ALL_TOKENS``: All tokens, including special tokens like [PAD], [EOS], [CLS], etc.

            - ``TOKEN``: Strictly tokens of the input.

            - ``WORD``: Tokens are grouped by word.

            - ``SENTENCE``: Tokens are grouped by sentence.

        Args:
            inputs_mapping (BatchEncoding): Tokenized inputs, the output of
                `self.tokenizer("some_text", return_tensors="pt", return_offsets_mapping=True)`
            granularity (Granularity | None, optional): Desired granularity level. Defaults to
                :attr:`DEFAULT`.
            tokenizer (PreTrainedTokenizer): Hugging-Face tokenizer used downstream.

        Raises:
            NoWordIdsError: if *WORD* granularity is requested with a slow
                            tokenizer.
            NotImplementedError: if an unknown granularity is supplied.

        Examples:
            >>> from interpreto.commons.granularity import Granularity
            >>> raw_input_text = [
            ...     "Interpreto is magical. Or is it?",
            ...     "At least we try.",
            ... ]
            >>> input_text_with_special_tokens = [
            ...     "[CLS]|Inter|preto| is| magic|al|.| Or| is| it|?|[EOS]",
            ...     "[CLS]|At| least| we| try|.|[EOS]|[PAD]|[PAD]|[PAD]|[PAD]|[PAD]",
            ... ]
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> input_ids = tokenizer(raw_input_text, return_tensors="pt")["input_ids"]
            >>> Granularity.get_indices(input_ids, granularity=Granularity.ALL_TOKENS, tokenizer=tokenizer)
            [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
             [[12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]]]
            >>> Granularity.get_indices(input_ids, granularity=Granularity.TOKEN, tokenizer=tokenizer)
            [[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
             [[13], [14], [15], [16], [17]]]
            >>> Granularity.get_indices(input_ids, granularity=Granularity.WORD, tokenizer=tokenizer)
            [[[1, 2], [3], [4, 5], [6], [7], [8], [9], [10]],
             [[13], [14], [15], [16], [17]]]
            >>> Granularity.get_indices(input_ids, granularity=Granularity.SENTENCE, tokenizer=tokenizer)
            [[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
             [[13, 14, 15, 16, 17]]]
        """

        match granularity or Granularity.DEFAULT:
            case Granularity.ALL_TOKENS:
                input_ids: Int[torch.Tensor, "n l"] = inputs["input_ids"]  # type: ignore
                return [Granularity.__all_tokens_get_indices(tokens_ids) for tokens_ids in input_ids]
            case Granularity.TOKEN:
                if tokenizer is None:
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is TOKEN."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                    )

                special_ids = tokenizer.all_special_ids
                input_ids: Int[torch.Tensor, "n l"] = inputs["input_ids"]  # type: ignore
                return [Granularity.__token_get_indices(tokens_ids, special_ids) for tokens_ids in input_ids]
            case Granularity.WORD:
                if tokenizer is None:
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is WORD."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                    )

                if not tokenizer.is_fast:
                    raise NoWordIdsError()

                n_inputs = inputs["input_ids"].shape[0]  # type: ignore
                return [Granularity.__word_get_indices(inputs.word_ids(i)) for i in range(n_inputs)]
            # spaCy-based levels (require offset_mapping & fast tokenizer)
            case Granularity.SENTENCE as level:
                if tokenizer is None:
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is TOKEN."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                    )

                # if not tokenizer or not tokenizer.is_fast:
                #     raise ValueError(f"{level.value} granularity needs a *fast* tokenizer.")
                if "offset_mapping" not in inputs:
                    raise ValueError(
                        f"{level.value} granularity requires `return_offsets_mapping=True` "
                        "when you call the tokenizer."
                    )

                if not _HAS_SPACY:
                    raise ModuleNotFoundError(
                        "spaCy is needed for sentence granularity.  Install with: `uv pip install spacy`"
                    )

                n_inputs = inputs["input_ids"].shape[0]  # type: ignore
                offset_maps = inputs["offset_mapping"]  # (n, lp, 2)

                return [
                    Granularity.__spacy_get_indices(
                        input_ids=inputs["input_ids"][i],  # type: ignore
                        offsets=offset_maps[i],  # type: ignore
                        tokenizer=tokenizer,
                        level=level,
                    )
                    for i in range(n_inputs)
                ]
            case _:
                raise NotImplementedError(f"Granularity level {granularity} not implemented")

    @staticmethod
    def __all_tokens_get_indices(tokens_ids) -> list[list[int]]:
        """Indices for :pyattr:`ALL_TOKENS` – every position kept."""
        length = len(tokens_ids)
        return [[i] for i in range(length)]

    @staticmethod
    def __token_get_indices(tokens_ids, special_ids) -> list[list[int]]:
        """Indices for :pyattr:`TOKEN` – skip special tokens."""
        return [[i] for i, tok_id in enumerate(tokens_ids) if tok_id not in special_ids]

    @staticmethod
    def __word_get_indices(word_ids) -> list[list[int]]:
        """Indices for :pyattr:`WORD` – group tokens belonging to the same word."""
        mapping: dict[int, list[int]] = {}
        for idx, wid in enumerate(word_ids):
            if wid is None:  # `None` for special tokens – ignore them
                continue
            mapping.setdefault(wid, []).append(idx)

        # Return groups ordered by word id (i.e. sentence order)
        return [mapping[k] for k in sorted(mapping)]

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def get_association_matrix(
        inputs: BatchEncoding,
        granularity: Granularity | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> list[Bool[torch.Tensor, "g lp"]]:
        """
        Creates the matrix to pass from one granularity level to ALL_TOKENS granularity level (finally used by the perturbator)

        Args:
            inputs (BatchEncoding): Tokenized inputs, the output of `self.tokenizer("some_text", return_tensors="pt", return_offsets_mapping=True)`
            granularity (Granularity | None, optional): Desired granularity level. Defaults to
                :attr:`DEFAULT`.
            tokenizer (PreTrainedTokenizer): Hugging-Face tokenizer used downstream.

        Raises:
            NotImplementedError: if granularity level is unknown, raises NotImplementedError

        Returns:
            list[torch.Tensor]: the list of matrices used to transform a specific granularity mask to a general mask that can be used on tokens.
                The list has ``n`` elements, each element is of shape ``(g, lp)``
                    ``g`` is the padded sequence length in the specific granularity,
                    and ``lp`` is the padded sequence length.
        """
        # get indices correspondence between granularity and ALL_TOKENS
        indices_list: list[list[list[int]]] = Granularity.get_indices(inputs, granularity, tokenizer)

        # iterate over the samples
        assoc_matrix_list: list[Bool[torch.Tensor, g, lp]] = []
        for indices in indices_list:
            g = len(indices)
            lp = inputs["input_ids"].shape[1]  # type: ignore

            # set to true matching positions in the matrix
            assoc_matrix: Bool[torch.Tensor, g, lp] = torch.zeros((g, lp), dtype=torch.bool)
            for j, gran_indices in enumerate(indices):
                assoc_matrix[j, gran_indices] = True
            assoc_matrix_list.append(assoc_matrix)

        return assoc_matrix_list

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def get_decomposition(
        inputs: BatchEncoding,
        granularity: Granularity | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        return_text: bool = False,
    ) -> list[list[list[int]]] | list[list[str]]:
        """
        Returns the token decomposition at the requested granularity level.
        Thus the a list of list of token indices is returned.

        This method groups token ids according to the chosen granularity. It can
        either keep every token, ignore special tokens or merge tokens that
        belong to the same word.

        Args:
            inputs (BatchEncoding): Tokenized inputs to decompose, the output of
                `self.tokenizer("some_text", return_tensors="pt", return_offsets_mapping=True)`
            granularity (Granularity | None, optional): Desired granularity level. Defaults to
                :attr:`DEFAULT`.
            tokenizer (PreTrainedTokenizer): Huggingface tokenizer used downstream.
            return_text (bool, optional): If True, the text corresponding to the token indices is returned.

        Returns:
            list[list[int]]: A nested list where the first level
                indexes the batch elements, the second level corresponds to groups of
                tokens and the last level contains the token ids inside each group.

        Raises:
            ValueError: If the tokenizer is not provided and return_text is True.
        """
        if not tokenizer and return_text:
            raise ValueError(
                "Tokenizer must be provided if return_text is True. Please provide a PreTrainedTokenizer instance."
            )

        # get indices correspondence between granularity and ALL_TOKENS
        indices_list = Granularity.get_indices(inputs, granularity, tokenizer)

        all_decompositions: list[list] = []
        for i, indices in enumerate(indices_list):
            input_ids: Int[torch.Tensor, "l"] = inputs["input_ids"][i]  # type: ignore
            # convert indices to token ids
            decomposition: list = []
            for gran_indices in indices:
                ids = [int(input_ids[idx].item()) for idx in gran_indices]

                if return_text:
                    text = tokenizer.decode(ids, skip_special_tokens=granularity is not Granularity.ALL_TOKENS)  # type: ignore
                    decomposition.append(text)
                else:
                    decomposition.append(ids)
            all_decompositions.append(decomposition)

        return all_decompositions

    @staticmethod
    @lru_cache(maxsize=2)  # keep a model in cache to reuse easily
    def __get_spacy(model: str = "en_core_web_sm"):
        """
        Lazily load a small spaCy pipeline.
        The model name can be patched via `SPACY_MODEL` env-var if needed.
        """

        try:
            nlp = spacy.load(model, disable=["ner", "tagger", "lemmatizer"])  # type: ignore
        except OSError as e:
            raise ModuleNotFoundError(
                "Unable to load spaCy model. Please download it via `python -m spacy download en_core_web_sm`"
            ) from e

        # sentence boundaries
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    @staticmethod
    def __spacy_get_indices(input_ids, offsets, tokenizer, level) -> list[list[int]]:
        """
        Generic spaCy-based grouper turning char-span segments (sent/para)
        into token-index groups.
        """

        # Build raw text (special tokens removed to keep offsets aligned)
        text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Run spaCy once per sample
        nlp = Granularity.__get_spacy(os.environ.get("SPACY_MODEL", "en_core_web_sm"))
        doc = nlp(text)

        # Obtain character spans for each requested granularity
        span_list = list(doc.sents)

        # Map char spans → token indices using the HF offset mapping
        groups: list[list[int]] = []
        for span in span_list:
            token_indices = [
                i
                for i, (s, e) in enumerate(offsets)
                if s is not None and e is not None and s >= span.start_char and e <= span.end_char + 1
            ]
            if token_indices:  # skip empty groups (can happen on only-punct spans)
                groups.append(token_indices)
        return groups

    def aggregate_subscores(token_scores: torch.tensor, granularity_method_aggregation: GranularityMethodAggregation):
        """
        Aggregates a set of token-level scores using the specified aggregation method.

        This function supports various strategies to combine multiple scores (e.g., across tokens
        corresponding to the same word) into a single value or vector. It handles both 1D and 2D tensors.

        Args:
            token_scores (torch.Tensor): A tensor containing scores to be aggregated.
                - If 1D: shape (num_tokens (for the world),)
                - If 2D: shape (num_scores, num_tokens)
            granularity_method_aggregation (GranularityMethodAggregation): The aggregation method to apply.
                Can be one of:
                    - MEAN: average of scores
                    - MAX: maximum score
                    - MIN: minimum score
                    - SUM: sum of scores
                    - SIGNED_MAX_ABS: score with the largest absolute value, preserving its sign

        Returns:
            torch.Tensor: The aggregated score(s).
                - If input is 1D: returns a single value (scalar).
                - If input is 2D: returns a 1D tensor of shape (num_scores,).

        Raises:
            NotImplementedError: If the aggregation method is not recognized.
        """
        match granularity_method_aggregation:
            case GranularityMethodAggregation.MEAN:
                return token_scores.mean(dim=0)
            case GranularityMethodAggregation.MAX:
                return token_scores.max(dim=0).values
            case GranularityMethodAggregation.MIN:
                return token_scores.min(dim=0).values
            case GranularityMethodAggregation.SUM:
                return token_scores.sum(dim=0)
            case GranularityMethodAggregation.SIGNED_MAX_ABS:
                if token_scores.dim() == 1:
                    max_idx = torch.argmax(token_scores.abs())
                    return token_scores[max_idx]
                else:
                    abs_token_scores = token_scores.abs()
                    max_indices = abs_token_scores.argmax(dim=0)
                    selected_scores = token_scores[max_indices, torch.arange(token_scores.shape[1])]
                    return selected_scores
            case _:
                raise NotImplementedError(f"Unknown aggregation method: {granularity_method_aggregation}")

    def aggregate_score_for_gradient_method(
        scores: torch.Tensor,
        granularity: Granularity | None,
        granularity_method_aggregation: GranularityMethodAggregation,
        inputs: BatchEncoding,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Aggregate scores for gradient-based methods according to the specified granularity.

        Args:
            scores (torch.Tensor): The scores to aggregate. Shape: (n, lp)
            granularity (Granularity | None): The granularity level to use for aggregation.
                If None, defaults to Granularity.DEFAULT.
            granularity_method_aggregation (GranularityMethodAggregation): The aggregation method to use.
            inputs (BatchEncoding, optional): Required for WORD-level aggregation.
            tokenizer (PreTrainedTokenizer, optional): Required for TOKEN/WORD-level filtering.

        Returns:
            torch.Tensor: The aggregated scores.
        """
        match granularity or Granularity.DEFAULT:
            case Granularity.ALL_TOKENS:
                return scores
            case Granularity.TOKEN:
                # ne doit renvoyer les scores que des "vrais" tokens
                if tokenizer is None:
                    raise ValueError("Tokenizer is required for TOKEN granularity.")
                special_ids = tokenizer.all_special_ids
                mask = torch.tensor(
                    [tok_id not in special_ids for tok_id in inputs["input_ids"][0]],
                    dtype=torch.bool,
                    device=scores.device,
                )
                if scores.dim() == 1:  # one score per token
                    return scores[mask]
                else:  # n scores per token
                    return scores[:, mask]
            case Granularity.WORD:
                if tokenizer is None or inputs is None:
                    raise ValueError("Tokenizer and inputs are required for WORD granularity.")
                if not tokenizer.is_fast:
                    raise NoWordIdsError()
                word_ids = inputs.word_ids(0)  # batch size = 1
                mapping: dict[int, list[int]] = {}
                for idx, wid in enumerate(word_ids):
                    if wid is not None:
                        mapping.setdefault(wid, []).append(idx)

                aggregated_scores = []
                for indices in mapping.values():
                    if scores.dim() == 1:  # one score per token
                        token_scores = scores[indices]
                    else:  # n scores per token
                        scores_T = scores.T
                        token_scores = scores_T[indices]
                    aggregated_scores.append(
                        Granularity.aggregate_subscores(token_scores, granularity_method_aggregation)
                    )
                if scores.dim() == 1:  # one score per token
                    return torch.stack(aggregated_scores)
                else:  # n scores per token
                    return torch.stack(aggregated_scores).T
