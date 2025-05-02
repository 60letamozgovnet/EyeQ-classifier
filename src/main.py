import model
import get_path 
import marking
import onnx_test
import preprocessing
import os
import logging
import autodownload

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    config = get_path.get_config()
    DATA_DIR = config['DATA_DIR']
    SRC_DIR = config["SRC_DIR"]

    flag = True if input("Do you want to install dataset? (y/n): ").lower() == 'y' else False
    if flag:
        autodownload.download_dataset(DATA_DIR)

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

    logger.info("Train of model was complete\n")

    session_cpu = onnx_test.session_cpu
    session_gpu = onnx_test.session_gpu

    input_data = onnx_test.input_data
    input_name = session_cpu.get_inputs()[0].name
    
    cpu_latency = onnx_test.measure_latency(session_cpu, input_name, input_data)
    gpu_latency = onnx_test.measure_latency(session_gpu, input_name, input_data)

    logger.info(f"Средняя задержка CPU: {cpu_latency:.2f} ms")
    logger.info(f"Средняя задержка GPU: {gpu_latency:.2f} ms")