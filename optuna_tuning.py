import optuna
from ultralytics import YOLO

def objective(trial):
    model = YOLO("yolov8s.pt")
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    momentum = trial.suggest_uniform('momentum', 0.85, 0.95)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)

    results = model.train(data="rgb.yaml", lr0=lr, momentum=momentum, weight_decay=weight_decay,epochs=10)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
2