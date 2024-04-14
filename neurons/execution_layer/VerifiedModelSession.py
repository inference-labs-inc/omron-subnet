import json
import os
import time
import uuid

import bittensor as bt
import ezkl
import onnxruntime

dir_path = os.path.dirname(os.path.realpath(__file__))


class VerifiedModelSession:

    def __init__(self, public_inputs=[]):
        self.model_id = 0
        self.session_id = str(uuid.uuid4())
        self.model_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/network.onnx"
        )
        self.pk_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/pk.key"
        )
        self.vk_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/vk.key"
        )
        self.srs_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/kzg.srs"
        )
        self.circuit_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/model.compiled"
        )
        self.settings_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/settings.json"
        )
        self.sample_input_path = os.path.join(
            dir_path, f"../deployment_layer/model_{self.model_id}/input.json"
        )

        self.witness_path = os.path.join(
            dir_path, f"./temp/witness_{self.model_id}_{self.session_id}.json"
        )
        self.input_path = os.path.join(
            dir_path, f"./temp/input_{self.model_id}_{self.session_id}.json"
        )
        self.proof_path = os.path.join(
            dir_path, f"./temp/model_{self.model_id}_{self.session_id}.proof"
        )

        self.ort_session = onnxruntime.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )

        self.public_inputs = public_inputs

        self.py_run_args = ezkl.PyRunArgs()
        self.py_run_args.input_visibility = "public"
        self.py_run_args.output_visibility = "public"
        self.py_run_args.param_visibility = "fixed"

    # Run an ORT prediction
    def run_model(self):
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.batch_size = 1
        input_data = self.public_inputs

        if input_data is None:
            bt.logging.error("Input data is None")
            return
        outputs = self.ort_session.run(
            None, {self.input_name: [[[d] for d in input_data]]}
        )
        self.outputs = outputs

    # Generate the input.json file, which is used in witness generation
    def gen_input_file(self):
        input_data = [self.public_inputs]
        input_shapes = [[self.batch_size]]
        data = {"input_data": input_data, "input_shapes": input_shapes}

        dir_name = os.path.dirname(self.input_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(self.input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def gen_witness(self):

        ezkl.gen_witness(
            self.input_path,
            self.circuit_path,
            self.witness_path,
            self.vk_path,
            self.srs_path,
        )

    def gen_proof_file(self, proof_string):
        dir_name = os.path.dirname(self.proof_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(self.proof_path, "w", encoding="utf-8") as f:
            f.write(proof_string)
            f.close()

    def gen_proof(self):

        try:
            bt.logging.debug("Running ONNX")
            self.run_model()
            bt.logging.debug("Generating input file")
            self.gen_input_file()
            bt.logging.debug("Generating witness")
            self.gen_witness()
            bt.logging.debug("Generating proof")
            start_time = time.time()
            ezkl.prove(
                self.witness_path,
                self.circuit_path,
                self.pk_path,
                self.proof_path,
                "single",
                self.srs_path,
            )
            end_time = time.time()
            bt.logging.info(f"Proof generation took {end_time - start_time} seconds")

            with open(self.proof_path, "r", encoding="utf-8") as f:
                proof_content = f.read()

            return proof_content

        except Exception as e:
            bt.logging.error(f"An error occured: {e}")
            return f"An error occured on miner proof: {e}"

    def verify_proof(self):
        res = ezkl.verify(
            self.proof_path,
            self.settings_path,
            self.vk_path,
            self.srs_path,
        )

        return res

    def verify_proof_string(self, proof_string):
        if proof_string == None:
            return False
        self.gen_proof_file(proof_string)
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
