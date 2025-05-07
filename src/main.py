import model
import get_path 
import marking
import onnx_test
import preprocessing
import os
import logging
import autodownload
import torch
import onnxruntime as ort
import numpy as np
from unzip_all import main as unzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    config = get_path.get_config()
    DATA_DIR = config['DATA_DIR']
    SRC_DIR = config["SRC_DIR"]

    flag = True if input("Do you want to install dataset? (y/n): ").lower() == 'y' else False
    if flag:
        autodownload.download_dataset(DATA_DIR)
        unzip()

    input_dir = os.path.join(DATA_DIR, 'train')
    output_dir = os.path.join(DATA_DIR, "preprocessed")

    preprocessing.preprocessing(input_dir, output_dir)
    logger.info("Preprocessing was complete\n")

    df = marking.create_df()
    df = marking.remove_rows(df, output_dir)
    df = marking.marking_dataset(df)
    df.to_csv(os.path.join(DATA_DIR, 'marking.csv'), encoding='utf-8')
    logger.info("Marking was complete\n")

    device = model.device
    model_ = model.create_model()
    criterion, optimizer = model.create_crit_optim(model_) 
    model.train(model_, device, criterion, optimizer, "./best_model.pth")
    model.torch.onnx.export(
        model_, 
        model.dummy_input, 
        os.path.join(DATA_DIR, 'EyeQ.onnx'),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    quantized_model = torch.quantizationquantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.onnx.export(
        quantized_model,
        model.dummy_input,
        os.path.join(DATA_DIR, 'EyeQ_quantized.onnx'),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    logger.info("Train of model was complete\n")

    session_cpu = ort.InferenceSession('./EyeQ.onnx', providers=["CPUExecutionProvider"])
    session_gpu = ort.InferenceSession('./EyeQ.onnx', providers=["CUDAExecutionProvider"])

    session_cpu_quant = ort.InferenceSession('./EyeQ_quantized.onnx', providers=["CPUExecutionProvider"])
    session_gpu_quant = ort.InferenceSession('./EyeQ_quantized.onnx', providers=["CUDAExecutionProvider"])

    input_name = session_cpu.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    cpu_latency = onnx_test.measure_latency(session_cpu, input_name, input_data)
    gpu_latency = onnx_test.measure_latency(session_gpu, input_name, input_data)

    logger.info(f"Средняя задержка CPU: {cpu_latency:.2f} ms")
    logger.info(f"Средняя задержка GPU: {gpu_latency:.2f} ms")

    cpu_latency = onnx_test.measure_latency(session_cpu_quant, input_name, input_data)
    gpu_latency = onnx_test.measure_latency(session_gpu_quant, input_name, input_data)

    logger.info(f"Средняя задержка CPU для сжатой модели: {cpu_latency:.2f} ms")
    logger.info(f"Средняя задержка GPU для сжатой модели: {gpu_latency:.2f} ms")