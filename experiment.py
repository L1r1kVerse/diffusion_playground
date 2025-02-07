import torch
import mlflow
import mlflow.pytorch
import pandas as pd 
from DDPM import train
from torch.utils.tensorboard import SummaryWriter

class Experiment:
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn, optimizer, corruption_helper,
                 epochs, scheduler=None, early_stopping=None, device=None, log_dir=None, note=None, use_mlflow=True, mlflow_tracking_dir=None):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.corruption_helper = corruption_helper
        self.epochs = epochs
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = device
        self.log_dir = log_dir if log_dir else "runs/experiment"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.note = note  # Store additional notes about the experiment
        self.use_mlflow = use_mlflow  # Toggle MLflow logging
        self.mlflow_tracking_dir = mlflow_tracking_dir if mlflow_tracking_dir else "/content/drive/MyDrive/Colab Notebooks/Diffusion/diffusion_playground/mlruns"

    def run(self):
        """Runs the experiment, logs results to TensorBoard and optionally MLflow."""
        # Initialize a list to store the losses for CSV
        epoch_data = []

        if self.use_mlflow:
            mlflow.start_run()  # Start MLflow tracking

            # Log experiment parameters
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("optimizer", self.optimizer.__class__.__name__)
            mlflow.log_param("learning_rate", self.optimizer.param_groups[0]["lr"])
            if self.scheduler:
                mlflow.log_param("scheduler", self.scheduler.__class__.__name__)
            if self.note:
                mlflow.log_param("note", self.note)

        # Run training
        train_losses, val_losses = train(
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.loss_fn,
            self.optimizer,
            self.corruption_helper,
            self.epochs,
            scheduler=self.scheduler if self.scheduler else None,
            early_stopping=self.early_stopping if self.early_stopping else None,
            writer=self.writer,  # TensorBoard logging remains unchanged
            device=self.device
        )

        # Log metrics to MLflow and collect data for CSV
        if self.use_mlflow:
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                # Collect data for CSV
                epoch_data.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # Create a DataFrame from the collected data
            df = pd.DataFrame(epoch_data)

            # Save the DataFrame as a CSV file
            csv_path = f"{self.mlflow_tracking_dir}/losses.csv"
            df.to_csv(csv_path, index=False)

            # Log the CSV file as an artifact in MLflow
            mlflow.log_artifact(csv_path)

            # Log trained model
            mlflow.pytorch.log_model(self.model, "model")

            # Save TensorBoard logs in MLflow
            mlflow.log_artifacts(self.log_dir, artifact_path="tensorboard_logs")

            mlflow.end_run()  # End MLflow tracking

        self.writer.close()
        return train_losses, val_losses

class ExperimentManager:
    def __init__(self):
        self.experiments = []

    def add_experiment(self, **kwargs):
        experiment = Experiment(**kwargs)
        self.experiments.append(experiment)
        return experiment

    def run_all(self):
        results = {}
        for i, experiment in enumerate(self.experiments):
            print(f"Running Experiment {i + 1}...")
            train_losses, val_losses = experiment.run()
            results[f"Experiment_{i + 1}"] = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "note": experiment.note  # Include experiment notes in results
            }
        return results
