from viz.viz_support import demo_init, run_demo
from viz.config import get_parse_args
import tensorflow as tf

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    args = get_parse_args()
    net, frame_provider = demo_init(args)
    run_demo(net, frame_provider)

    # python run_demo.py --model_name 'model_oop_cnn' --images ../Data/safety_class_dataset/20211025_*_unbelt_nomask_jieun/Color/*.jpg
    # python run_demo.py --model_name 'model_oop_cnn' --images ../Data/safety_class_dataset/20211015_*_unbelt_nomask_sungwook*/Color/*.jpg
    # python run_demo.py --model_name 'model_oop_cnn' --video 0     