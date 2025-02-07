import sys
import numpy as np
import onnxruntime as ort
import traceback


def force_print(msg: str):
    print(msg, file=sys.stderr, flush=True)
    sys.stderr.flush()


def run_inference(model_path: str, input_path: str, output_path: str) -> None:
    try:
        force_print("\n### ONNX RUNNER STARTING ###")
        force_print(f"### Loading model from: {model_path}")
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        force_print(f"### Input tensor name: {input_name}")
        input_data = np.load(input_path)
        force_print(f"### Loaded input data shape: {input_data.shape}")

        options = ort.SessionOptions()
        options.log_severity_level = 3

        outputs = session.run(None, {input_name: input_data}, options)
        force_print(f"### Raw output shapes: {[out.shape for out in outputs]}")

        flattened = []
        for out in outputs:
            flattened.extend(out.flatten())
        final_output = np.array(flattened)
        force_print(f"### Final flattened output shape: {final_output.shape}")
        force_print(f"### Saving output to: {output_path}")
        np.save(output_path, final_output)
        force_print("### ONNX RUNNER COMPLETED ###\n")
    except Exception as e:
        force_print(f"Error running inference: {str(e)}")
        force_print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python onnx_runner.py <model_path> <input_path> <output_path>",
            file=sys.stderr,
        )
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
