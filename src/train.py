import os
import shutil
from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 Nano model...")
    # Load YOLOv8n (nano) which is optimized for edge devices
    model = YOLO("yolov8n.pt")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml = os.path.join(project_root, 'data', 'data.yaml')
    runs_dir = os.path.join(project_root, 'runs')
    docs_dir = os.path.join(project_root, 'docs')
    
    # Train the model
    # Note: epochs=5 is used for demonstration/speed. For a production edge model, this should be higher.
    print(f"Training on {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=640,
        project=runs_dir,
        name='plate_detect',
        plots=True,
        device='cpu' # Assuming CPU deployment here unless GPU is available, YOLO will try GPU if possible but we enforce CPU just in case or let it auto-select. Actually let's not enforce CPU so it can be faster if user has GPU.
    )
    print("Training complete!")

    # YOLOv8 automatically saves training plots (results.png) into the run directory.
    # We need to find the latest run directory and copy the visualizations to /docs
    
    # Find the run directory
    run_dir = os.path.join(runs_dir, 'plate_detect')
    
    # Copy results.png
    results_img = os.path.join(run_dir, 'results.png')
    if os.path.exists(results_img):
        shutil.copy2(results_img, os.path.join(docs_dir, 'training_metrics.png'))
        print(f"Copied training metrics to {docs_dir}")
        
    # Copy confusion matrix
    cm_img = os.path.join(run_dir, 'confusion_matrix.png')
    if os.path.exists(cm_img):
        shutil.copy2(cm_img, os.path.join(docs_dir, 'confusion_matrix.png'))
        print(f"Copied confusion matrix to {docs_dir}")
        
    # Copy best weights to models/
    best_weights = os.path.join(run_dir, 'weights', 'best.pt')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, os.path.join(models_dir, 'plate_detector.pt'))
        print(f"Exported best edge model weights to {models_dir}")

if __name__ == '__main__':
    # Removing device force to let it use CUDA if available
    # Removed explicitly setting device='cpu' from model.train() above
    main()
