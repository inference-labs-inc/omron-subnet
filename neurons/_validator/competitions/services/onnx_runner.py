import sys
import numpy as np
import onnxruntime as ort


def run_inference(model_path: str, input_path: str, output_path: str) -> None:
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_data = np.load(input_path)
        outputs = session.run(None, {input_name: input_data})[0]
        np.save(output_path, outputs)
    except Exception as e:
        print(f"Error running inference: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
