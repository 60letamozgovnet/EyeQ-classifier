import onnxruntime as ort
import numpy as np
import time

session_cpu = ort.InferenceSession(r'D:\code\python\ML_PIR\main\EyeQ.onnx', providers=["CPUExecutionProvider"])
session_gpu = ort.InferenceSession(r'D:\code\python\ML_PIR\main\EyeQ.onnx', providers=["CUDAExecutionProvider"])


input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

def measure_latency(session, input_name, input_data, runs=100):
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})

    start = time.time()
    for _ in range(runs):
        _ = session.run(None, {input_name: input_data})
    end = time.time()
    avg_time = (end - start) / runs * 1000
    return avg_time

input_name = session_cpu.get_inputs()[0].name

cpu_latency = measure_latency(session_cpu, input_name, input_data)
gpu_latency = measure_latency(session_gpu, input_name, input_data)

print(f"Средняя задержка CPU: {cpu_latency:.2f} ms")
print(f"Средняя задержка GPU: {gpu_latency:.2f} ms")