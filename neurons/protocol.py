"""
The MIT License (MIT)
Copyright © 2023 Chris Wilson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, Optional
from attr import field

import bittensor as bt

class ProvingSystem(Enum):
    TEE="TrustedExecution"
    ZK="ZeroKnowledge"

class QueryZkProof(bt.Synapse):
    """
    QueryZkProof class inherits from bt.Synapse.
    It is used to query zkproof of certain model.
    """

    # Required request input, filled by sending dendrite caller.
    query_input: Optional[Dict] = None

    # Optional request output, filled by receiving axon.
    query_output: Optional[str] = None


class QueryForProvenInference(bt.Synapse):
    """
    A Synapse for querying proven inferences.
    DEV: This synapse is a placeholder.
    """
    model_id: str = field(init=False)
    proof_system: ProvingSystem = field(init=False, default=ProvingSystem.ZK)
    query_input: Optional[Dict] = field(init=False)
    query_output: Optional[Dict] = field(init=False)


class QueryForProofAggregation(bt.Synapse):
    """
    Query for aggregation of multiple proofs into a single proof
    """

    proofs: list[str] = []
    model_id: str or int
    aggregation_proof: Optional[str] = None

    def deserialize(self) -> str:
        """
        Return the aggregation proof
        """
        return self.aggregation_proof
