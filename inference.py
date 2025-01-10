"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import time

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def run():
    loading_start_time = time.perf_counter()
    # Read the input
    input_frame_rate = load_json_file(
         location=INPUT_PATH / "frame-rate.json",
    )
    input_magnetic_field_strength = load_json_file(
         location=INPUT_PATH / "b-field-strength.json",
    )
    input_scanned_region = load_json_file(
         location=INPUT_PATH / "scanned-region.json",
    )
    input_mri_linac_series = load_image_file_as_array(
        location=INPUT_PATH / "images/mri-linacs",
    )
    input_mri_linac_target = load_image_file_as_array(
        location=INPUT_PATH / "images/mri-linac-target",
    )

    print(f"Loading took {time.perf_counter() - loading_start_time} seconds")
    # Process the inputs: any way you'd like
    #_show_torch_cuda_info()

    #with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
    #    print(f.read())

    # For now, let us make bogus predictions
    #output_mri_linac_series_targets = numpy.eye(4, 2)
    #print("input_mri_linac_series.shape", input_mri_linac_series.shape)
    #print("input_mri_linac_target.shape", input_mri_linac_target.shape)
    algo_start_time = time.perf_counter()
    # For the example we want to repeat the initial segmentation for every frame 
    output_mri_linac_series_targets = np.repeat(input_mri_linac_target, input_mri_linac_series.shape[2], axis=-1)#[:,None,:,:]
    #time.sleep(0.002*output_mri_linac_series_targets.shape[2])
    print(f"Algorithm took {time.perf_counter() - algo_start_time} seconds")
    #print(output_mri_linac_series_targets.shape)
    #(270, 270, 5)
    #print(output_mri_linac_series_targets.shape)

    # input_mri_linac_series.shape == (T, W, H)
    writing_start_time = time.perf_counter()
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/mri-linac-series-targets",
        array=output_mri_linac_series_targets,
    )
    print(f"Writing took {time.perf_counter() - writing_start_time} seconds")
    
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, 'r') as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())