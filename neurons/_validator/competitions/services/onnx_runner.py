import sys
import numpy as np
import onnxruntime as ort


def run_inference(model_path: str, input_path: str, output_path: str) -> None:
    session = ort.InferenceSession(model_path)
    input_data = np.load(input_path)
    outputs = session.run(None, {"input": input_data})[0]
    np.save(output_path, outputs)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
