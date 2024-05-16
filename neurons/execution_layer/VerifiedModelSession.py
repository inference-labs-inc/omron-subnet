import asyncio
import json
import multiprocessing
import os
import time
import uuid

import bittensor as bt
import ezkl

dir_path = os.path.dirname(os.path.realpath(__file__))


def gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path):
    bt.logging.debug("Generating witness")
    res = ezkl.gen_witness(
        input_path,
        circuit_path,
        witness_path,
        vk_path,
        srs_path,
    )
    bt.logging.debug(f"Gen witness result: {res}")


async def gen_proof(
    input_path, vk_path, witness_path, circuit_path, pk_path, proof_path, srs_path
):
    gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path)

    bt.logging.debug("Generating proof!!!!!!")
    ezkl.prove(
        witness_path,
        circuit_path,
        pk_path,
        proof_path,
        "single",
        srs_path,
    )


def proof_worker(
    input_path, vk_path, witness_path, circuit_path, pk_path, proof_path, srs_path
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            gen_proof(
                input_path,
                vk_path,
                witness_path,
                circuit_path,
                pk_path,
                proof_path,
                srs_path,
            )
        )
        return result
    finally:
        loop.close()


class VerifiedModelSession:

    def __init__(self, public_inputs=[]):
        self.model_id = 0
        self.session_id = str(uuid.uuid4())
        model_path = os.path.join(
            os.path.dirname(dir_path), f"deployment_layer/model_{self.model_id}"
        )

        self.pk_path = os.path.join(model_path, "pk.key")
        self.vk_path = os.path.join(model_path, "vk.key")
        self.srs_path = os.path.join(model_path, "kzg.srs")
        self.circuit_path = os.path.join(model_path, "model.compiled")
        self.settings_path = os.path.join(model_path, "settings.json")
        self.sample_input_path = os.path.join(model_path, "input.json")

        self.witness_path = os.path.join(
            dir_path, f"temp/witness_{self.model_id}_{self.session_id}.json"
        )
        self.input_path = os.path.join(
            dir_path, f"temp/input_{self.model_id}_{self.session_id}.json"
        )
        self.proof_path = os.path.join(
            dir_path, f"temp/model_{self.model_id}_{self.session_id}.proof"
        )

        self.public_inputs = public_inputs

        self.py_run_args = ezkl.PyRunArgs()
        self.py_run_args.input_visibility = "public"
        self.py_run_args.output_visibility = "public"
        self.py_run_args.param_visibility = "fixed"
        self.batch_size = 1

    # Generate the input.json file, which is used in witness generation
    def gen_input_file(self):
        input_data = [self.public_inputs]
        input_shapes = [[self.batch_size]]
        data = {"input_data": input_data, "input_shapes": input_shapes}

        dir_name = os.path.dirname(self.input_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(self.input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def gen_proof_file(self, proof_string, instances):
        dir_name = os.path.dirname(self.proof_path)
        os.makedirs(dir_name, exist_ok=True)
        proof_json = json.loads(proof_string)
        new_instances = instances[0]
        bt.logging.trace(f"New instances: {new_instances}")
        new_instances.append(proof_json["instances"][0][-1])
        bt.logging.trace(
            f"New instances after appending with last instance from output: {new_instances}"
        )
        proof_json["instances"] = [new_instances]
        # Enforce EVM transcript type to be used for all proofs, ensuring all are valid for proving on EVM chains
        proof_json["transcript_type"] = "EVM"
        bt.logging.trace(f"Proof json: {proof_json}")

        with open(self.proof_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(proof_json))
            f.close()

    def gen_proof(self):

        try:
            bt.logging.debug("Generating input file")
            self.gen_input_file()
            bt.logging.debug("Starting generating proof process...")
            start_time = time.time()
            with multiprocessing.Pool(1) as p:
                p.apply(
                    func=proof_worker,
                    kwds={
                        "input_path": self.input_path,
                        "vk_path": self.vk_path,
                        "witness_path": self.witness_path,
                        "circuit_path": self.circuit_path,
                        "pk_path": self.pk_path,
                        "proof_path": self.proof_path,
                        "srs_path": self.srs_path,
                    },
                )

            end_time = time.time()
            proof_time = end_time - start_time
            bt.logging.info(f"Proof generation took {proof_time} seconds")
            with open(self.proof_path, "r", encoding="utf-8") as f:
                proof_content = f.read()

            return proof_content, proof_time

        except Exception as e:
            bt.logging.error(f"An error occured: {e}")
            return f"An error occured on miner proof: {e}", 0

    def verify_proof(self):
        res = ezkl.verify(
            self.proof_path,
            self.settings_path,
            self.vk_path,
            self.srs_path,
        )

        return res

    def verify_proof_and_inputs(self, proof_string, inputs):
        if proof_string == None:
            return False
        self.public_inputs = inputs
        self.gen_input_file()
        gen_witness(
            self.input_path,
            self.circuit_path,
            self.witness_path,
            self.vk_path,
            self.srs_path,
        )
        with open(self.witness_path, "r", encoding="utf-8") as f:
            witness_content = f.read()
        witness_json = json.loads(witness_content)
        self.gen_proof_file(proof_string, witness_json["inputs"])
        return self.verify_proof()

    def __enter__(self):
        return self

    def remove_temp_files(self):
        if os.path.exists(self.input_path):
            os.remove(self.input_path)

        if os.path.exists(self.witness_path):
            os.remove(self.witness_path)

        if os.path.exists(self.proof_path):
            os.remove(self.proof_path)

    def end(self):
        self.remove_temp_files()

    def __exit__(self, exc_type, exc_val, exc_tb):

        return None
