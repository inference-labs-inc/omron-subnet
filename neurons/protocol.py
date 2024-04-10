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

from typing import Dict, Optional

import bittensor as bt


class QueryZkProof(bt.Synapse):
    """
    QueryZkProof class inherits from bt.Synapse.
    It is used to query zkproof of certain model.
    """

    # Required request input, filled by sending dendrite caller.
    query_input: Optional[Dict] = None

    # Optional request output, filled by receiving axon.
    query_output: Optional[str] = None

    def deserialize(self: bt.Synapse) -> str:
        """
        unpack query_output
        """
        return self.query_output


class CheckMiner(bt.Synapse):
    """
    CheckMiner class inherits from bt.Synapse.
    It is used to check the miner's status. This is not used yet.
    """

    # Required request input, send url_hash for check
    check_url_hash: str

    # TODO: Add error handling for when check_output is None
    check_output: Optional[Dict] = None

    def deserialize(self) -> Dict:
        """
        Deserialize the check_output into a dictionary.
        """
        # TODO: Add error handling for when check_output is None
        return self.check_output
