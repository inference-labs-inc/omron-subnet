import sys
import numpy as np
import onnxruntime as ort


def run_inference(model_path: str, input_path: str, output_path: str) -> None:
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_data = np.load(input_path)

        output_names = [output.name for output in session.get_outputs()]
        outputs = session.run(output_names, {input_name: input_data})

        flattened = []
        for out in outputs:
            flattened.extend(out.flatten())
        np.save(output_path, np.array(flattened))
    except Exception as e:
        print(f"Error running inference: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
