import onnxruntime as ort
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def measure_latency(session, input_name, input_data, runs=100):
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})

    start = time.time()
    for _ in range(runs):
        _ = session.run(None, {input_name: input_data})
    end = time.time()
    avg_time = (end - start) / runs * 1000
    return avg_time



if __name__ == "__main__":
    session_cpu = ort.InferenceSession('./EyeQ.onnx', providers=["CPUExecutionProvider"])
    session_gpu = ort.InferenceSession('./EyeQ.onnx', providers=["CUDAExecutionProvider"])

    session_cpu_quant = ort.InferenceSession('./EyeQ_quantized.onnx', providers=["CPUExecutionProvider"])
    session_gpu_quant = ort.InferenceSession('./EyeQ_quantized.onnx', providers=["CUDAExecutionProvider"])

    input_name = session_cpu.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    cpu_latency = measure_latency(session_cpu, input_name, input_data)
    gpu_latency = measure_latency(session_gpu, input_name, input_data)

    logger.info(f"Средняя задержка CPU: {cpu_latency:.2f} ms")
    logger.info(f"Средняя задержка GPU: {gpu_latency:.2f} ms")

    cpu_latency = measure_latency(session_cpu_quant, input_name, input_data)
    gpu_latency = measure_latency(session_gpu_quant, input_name, input_data)

    logger.info(f"Средняя задержка CPU для сжатой модели: {cpu_latency:.2f} ms")
    logger.info(f"Средняя задержка GPU для сжатой модели: {gpu_latency:.2f} ms")
