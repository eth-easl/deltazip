import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
from deltazip.pipelines.utils import initialize

initialize()


def profile_disk_io(
    test_file=".cache/compressed_models/bits-3/llama-2-7b-chat/deltazip-compressed.safetensors",
):
    # run the command and get output
    subprocess.check_output(
        "sudo echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True
    )

    output = subprocess.run(
        "time dd if=.cache/compressed_models/bits-2/llama-2-7b-chat/deltazip-compressed.safetensors of=/dev/null bs=8k",
        shell=True,
        stderr=subprocess.PIPE,
    )

    output = str(output.stderr)
    output = output.split(" s,")[1].split("/s")[0].strip()
    if "GB" in output:
        return output
    else:
        raise Exception("Output is not in GB/s")


def get_gpu_name():
    handle = nvmlDeviceGetHandleByIndex(0)
    name = nvmlDeviceGetName(handle)
    return name


if __name__ == "__main__":
    profile_disk_io()
