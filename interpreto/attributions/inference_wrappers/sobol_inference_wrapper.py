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

from __future__ import annotations

import torch


class SobolInferenceWrapper:
    def __init__(self, model):
        self.model = model

    def inference(self, inputs_model):
        """
        Run the model on the inputs_model and return the logits.
        """
        with torch.no_grad():
            outputs = self.model(**inputs_model)
        return outputs.logits

    def batched_inference(self, inputs_model, batch_size=32):
        """
        Run the model on the inputs_model in batches and return the concatenated logits.

        Parameters:
        - inputs_model: dictionary with keys "input_ids" and "attention_mask", each of shape (n_samples, seq_len)
        - batch_size: the batch size for inference.
        """
        input_ids = inputs_model["input_ids"]
        attention_mask = inputs_model["attention_mask"]
        n_samples = input_ids.shape[0]
        logits_list = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_input_ids = input_ids[i : i + batch_size]
                batch_attention_mask = attention_mask[i : i + batch_size]
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits_list.append(outputs.logits)
        logits = torch.cat(logits_list, dim=0)
        return logits
