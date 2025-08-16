import ray
from ray import tune
from ultralytics import YOLO

RAY_memory_monitor_refresh_ms=0

def train_yolo(config):
    try:
        model = YOLO("yolov8s.pt")
        results = model.train(
            data="/home/arjun/Desktop/icpr/rgb.yaml",
            epochs=10,
            **config
        )
        return results
    except Exception as e:
        print(f"Error in train_yolo: {str(e)}")
        raise

# Initialize Ray
ray.init(num_gpus=1)

# Updated search space with valid YOLO arguments
search_space = {
    "lr0": tune.loguniform(1e-5, 1e-1),
    "lrf": tune.uniform(0.01, 1.0),
    "momentum": tune.uniform(0.6, 0.98),
    "weight_decay": tune.loguniform(1e-5, 1e-1),
    "warmup_epochs": tune.uniform(0, 5),
    "warmup_momentum": tune.uniform(0, 0.95),
    "box": tune.uniform(0.02, 0.2),
    "cls": tune.uniform(0.2, 4.0),
    "iou": tune.uniform(0.1, 0.7),
    "hsv_h": tune.uniform(0.0, 0.1),
    "hsv_s": tune.uniform(0.0, 0.9),
    "hsv_v": tune.uniform(0.0, 0.9),
    "degrees": tune.uniform(0.0, 45.0),
    "translate": tune.uniform(0.0, 0.9),
    "scale": tune.uniform(0.0, 0.9),
    "shear": tune.uniform(0.0, 10.0),
    "perspective": tune.uniform(0.0, 0.001),
    "flipud": tune.uniform(0.0, 1.0),
    "fliplr": tune.uniform(0.0, 1.0),
    "mosaic": tune.uniform(0.0, 1.0),
    "mixup": tune.uniform(0.0, 1.0),
    "copy_paste": tune.uniform(0.0, 1.0),
}

# Run the hyperparameter tuning
try:
    analysis = tune.run(
        train_yolo,
        config=search_space,
        num_samples=3,
        resources_per_trial={"gpu": 1,"cpu" : 12},
        metric="metrics/mAP50-95(B)",
        mode="max",
        max_failures=3  # Allow retries if tasks fail
    )

    # Get the best configuration
    best_config = analysis.get_best_config(metric="metrics/mAP50-95(B)", mode="max")
    print("Best hyperparameters found:", best_config)
except Exception as e:
    print(f"Error during tuning: {str(e)}")
