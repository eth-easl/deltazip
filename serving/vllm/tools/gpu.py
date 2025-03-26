import xml.etree.ElementTree as ET
import subprocess
def parse_nvidia_smi(xml_output):
    tree = ET.ElementTree(ET.fromstring(xml_output))
    root = tree.getroot()
    
    # To store processes on GPU 2 and 3
    gpu_processes = {}

    # Loop over GPUs
    for gpu in root.findall("gpu"):
        gpu_id = int(gpu.find("minor_number").text)
        
        if gpu_id in [2, 3]:
            processes = gpu.find("processes")
            running_processes = []
            
            if processes is not None:
                for process in processes.findall("process_info"):
                    pid = process.find("pid").text
                    name = process.find("process_name").text
                    used_memory = process.find("used_memory").text
                    running_processes.append({
                        "pid": pid,
                        "name": name,
                        "used_memory": used_memory
                    })
                    
            gpu_processes[gpu_id] = running_processes
    
    return gpu_processes

def get_processes():
    sp = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    return parse_nvidia_smi(out_str[0])

if __name__=="__main__":
    import os
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    print(parse_nvidia_smi(out_str[0]))