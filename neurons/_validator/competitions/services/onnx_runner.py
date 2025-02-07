import sys
import numpy as np
import onnxruntime as ort
import traceback
import os

os.environ["ONNXRUNTIME_LOGGING_LEVEL"] = "3"
ort.set_default_logger_severity(3)


def run_inference(model_path: str, input_path: str, output_path: str) -> None:
    try:
        print("\n=== ONNX RUNNER STARTING ===")
        print(f"Loading model from: {model_path}")
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        print(f"Input tensor name: {input_name}")
        input_data = np.load(input_path)
        print(f"Loaded input data shape: {input_data.shape}")

        outputs = session.run(None, {input_name: input_data})
        print(f"Raw output shapes: {[out.shape for out in outputs]}")

        flattened = []
        for out in outputs:
            flattened.extend(out.flatten())
        final_output = np.array(flattened)
        print(f"Final flattened output shape: {final_output.shape}")
        print(f"Saving output to: {output_path}")
        np.save(output_path, final_output)
        print("=== ONNX RUNNER COMPLETED ===\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error running inference: {str(e)}", file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python onnx_runner.py <model_path> <input_path> <output_path>",
            file=sys.stderr,
        )
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
