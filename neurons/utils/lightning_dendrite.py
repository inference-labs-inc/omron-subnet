import time
from typing import Union, List
import bittensor as bt
from bittensor.core.settings import version_as_int
from utils.lightning import lightning


class LightningDendrite:
    """
    Drop-in replacement for bittensor dendrite using Rust for true parallelism.
    """

    def __init__(self, wallet: bt.wallet, external_ip: str = None):

        self.wallet = wallet
        self.external_ip = external_ip or "127.0.0.1"
        self.uuid = str(time.time_ns())

        self.rust_dendrite = lightning.LightningDendrite(
            wallet_hotkey=wallet.hotkey.ss58_address, external_ip=self.external_ip
        )

    async def call(
        self,
        target_axon: Union[bt.AxonInfo, bt.Axon],
        synapse: bt.Synapse,
        timeout: float = 60.0,
        deserialize: bool = True,
        **kwargs,
    ) -> bt.Synapse:
        """
        Asynchronously send request to a single axon using Rust backend.
        """
        start_time = time.time()

        if isinstance(target_axon, bt.Axon):
            target_axon = target_axon.info()

        synapse = self._preprocess_synapse_for_request(target_axon, synapse, timeout)

        signature = self._sign_synapse(synapse)

        axon_dict = {
            "ip": target_axon.ip,
            "port": target_axon.port,
            "hotkey": target_axon.hotkey,
        }

        synapse_headers = synapse.to_headers()
        synapse_body = synapse.model_dump()

        result = self.rust_dendrite.call(
            target_axon=axon_dict,
            synapse_headers=synapse_headers,
            synapse_body=synapse_body,
            signature=signature,
            timeout=timeout,
        )

        self._process_rust_response(result, synapse)
        synapse.dendrite.process_time = str(time.time() - start_time)

        return synapse.deserialize() if deserialize else synapse

    async def forward(
        self,
        axons: Union[List[bt.AxonInfo], List[bt.Axon]],
        synapse: bt.Synapse,
        timeout: float = 60.0,
        deserialize: bool = True,
        **kwargs,
    ) -> List[bt.Synapse]:
        """
        Asynchronously send requests to multiple axons using Rust backend.
        """

        axon_infos = []
        for axon in axons:
            if isinstance(axon, bt.Axon):
                axon_infos.append(axon.info())
            else:
                axon_infos.append(axon)

        processed_synapses = []
        signatures = []
        axon_dicts = []

        for axon_info in axon_infos:

            synapse_copy = synapse.model_copy()
            synapse_copy = self._preprocess_synapse_for_request(
                axon_info, synapse_copy, timeout
            )
            signature = self._sign_synapse(synapse_copy)

            processed_synapses.append(synapse_copy)
            signatures.append(signature)
            axon_dicts.append(
                {
                    "ip": axon_info.ip,
                    "port": axon_info.port,
                    "hotkey": axon_info.hotkey,
                }
            )

        synapse_headers = processed_synapses[0].to_headers()
        synapse_body = processed_synapses[0].model_dump()

        results = self.rust_dendrite.forward(
            axons=axon_dicts,
            synapse_headers=synapse_headers,
            synapse_body=synapse_body,
            signatures=signatures,
            timeout=timeout,
        )

        final_results = []
        for i, (result, synapse_copy) in enumerate(zip(results, processed_synapses)):
            self._process_rust_response(result, synapse_copy)

            if deserialize:
                final_results.append(synapse_copy.deserialize())
            else:
                final_results.append(synapse_copy)

        return final_results

    def _preprocess_synapse_for_request(
        self,
        target_axon_info: bt.AxonInfo,
        synapse: bt.Synapse,
        timeout: float,
    ) -> bt.Synapse:
        """Preprocess synapse for making a request."""
        synapse.timeout = timeout
        synapse.dendrite = bt.TerminalInfo(
            ip=self.external_ip,
            version=version_as_int,
            nonce=time.time_ns(),
            uuid=self.uuid,
            hotkey=self.wallet.hotkey.ss58_address,
        )

        synapse.axon = bt.TerminalInfo(
            ip=target_axon_info.ip,
            port=target_axon_info.port,
            hotkey=target_axon_info.hotkey,
        )

        return synapse

    def _sign_synapse(self, synapse: bt.Synapse) -> str:
        """Sign the synapse request."""
        # flake8: noqa
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
        signature = f"0x{self.wallet.hotkey.sign(message).hex()}"
        synapse.dendrite.signature = signature
        return signature

    def _process_rust_response(self, rust_response: dict, synapse: bt.Synapse):
        """Process response from Rust and update synapse."""

        synapse.dendrite.status_code = rust_response.get("status_code", "500")
        synapse.dendrite.status_message = rust_response.get(
            "status_message", "Unknown error"
        )
        synapse.dendrite.process_time = rust_response.get("process_time", "0.0")

        synapse.axon.status_code = synapse.dendrite.status_code
        synapse.axon.status_message = synapse.dendrite.status_message

        response_data = rust_response.get("response_data")
        if response_data and synapse.dendrite.status_code == "200":
            try:
                import json

                json_response = json.loads(response_data)

                server_synapse = synapse.__class__(**json_response)

                for key in synapse.model_dump().keys():
                    try:
                        setattr(synapse, key, getattr(server_synapse, key))
                    except Exception:
                        pass

            except Exception:
                pass

    def __str__(self) -> str:
        return f"lightning_dendrite({self.wallet.hotkey.ss58_address})"

    def __repr__(self) -> str:
        return self.__str__()
