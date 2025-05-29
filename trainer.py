import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import operator

class Trainer:
    """
    Trainer class for training, validating, and testing a PyTorch model.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
    """
    mode_dict = {"min": operator.lt, "max": operator.gt}

    def __init__(self,
            model,
            device,
        ):
        self.model = model
        self.device = device

    def fit(self,
            train_loader,
            optimizer,
            criterion,
            max_epochs,
            early_stopping=False,
            patience=5,
            val_loader=None,
            test_loader=None,
            early_stopping_monitor='loss', 
            early_stopping_mode='min',
            scheduler=None,
            metrics: dict = {},
        ):
        """
        Trains the model using the provided data loaders, optimizer, and criterion.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (torch.nn.Module): Loss function.
            max_epochs (int): Maximum number of epochs to train.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 5.
            val_loader (DataLoader, optional): DataLoader for the validation data. Defaults to None.
            test_loader (DataLoader, optional): DataLoader for the test data. Defaults to None.
            early_stopping_monitor (str, optional): Metric to monitor for early stopping. Defaults to 'loss'.
            early_stopping_mode (str, optional): Mode for early stopping ('min' or 'max'). Defaults to 'min'.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
            metrics (dict, optional): Dictionary of metrics to compute. Defaults to {}.

        Returns:
            dict: Dictionary containing training, validation, and test losses and metrics.
        """
        def print_table_header():
            base = f"| {'Epoch':<6}| {'lr':<10}| {'train_loss':<11}|"
            val_loss_str = f" {'val_loss':<11}|" if val_loader is not None and val_loss is not None else ""

            monitor_str = ""
            if early_stopping and early_stopping_monitor != 'loss':
                monitor_str = f" {f'train_{early_stopping_monitor}':<20}| {f'val_{early_stopping_monitor}':<20}|"

            time_str = f" {'time (s)':<10}|"

            header = base + val_loss_str + monitor_str + time_str
            header_len = len(header)
            
            print()
            print("-"*header_len)
            print(header)
            print("-"*header_len)

            return header_len

        def print_table_row():
            elapsed_time = end_time - start_time

            base_row = f"| {epoch:<6}| {lr:<10.3E}| {train_loss_epoch:<11.3E}|"
            
            val_loss_str = f" {val_loss_epoch:<11.3E}|" if val_loader is not None else ""
            
            monitor_str = ""
            if early_stopping and early_stopping_monitor != 'loss':
                train_val = train_metrics_epoch[early_stopping_monitor]
                val_val = val_metrics_epoch[early_stopping_monitor]
                monitor_str = f" {train_val:<20.4f}| {val_val:<20.4f}|"
            
            time_str = f" {elapsed_time:<10.3f}|" if elapsed_time is not None else ""
            early_stop_flag = " (!) Early stopping" if early_stopping_reached else ""
            
            print(base_row + val_loss_str + monitor_str + time_str + early_stop_flag)

        train_loss = []
        val_loss = []
        test_loss = []

        train_metrics = {f'train_{metric}': [] for metric in metrics.keys()}
        val_metrics = {f'val_{metric}': [] for metric in metrics.keys()}
        test_metrics = {f'test_{metric}': [] for metric in metrics.keys()}

        learning_rates = []

        best_val_monitor = float('inf') if early_stopping_mode == 'min' else float('-inf')
        best_model = None
        patience_counter = 0
        early_stopping_reached = False

        header_len = print_table_header()

        try:
            for epoch in range(max_epochs):
                if early_stopping_reached:
                    break

                start_time = time.time()
                
                lr = optimizer.param_groups[0]['lr']
                learning_rates.append(lr)

                train_loss_epoch, train_metrics_epoch = self._train_epoch(train_loader, optimizer, criterion, metrics)
                train_loss.append(train_loss_epoch)
                
                if val_loader is not None:
                    val_loss_epoch, val_metrics_epoch = self._validate_epoch(val_loader, criterion, metrics, task='Evaluation')
                    val_loss.append(val_loss_epoch)
                
                if test_loader is not None:
                    test_loss_epoch, test_metrics_epoch = self._validate_epoch(test_loader, criterion, metrics, task='Test Evaluation')
                    test_loss.append(test_loss_epoch)

                for metric in metrics.keys():
                    train_metrics[f'train_{metric}'].append(train_metrics_epoch[metric])
                    if val_loader is not None:
                        val_metrics[f'val_{metric}'].append(val_metrics_epoch[metric])
                    if test_loader is not None:
                        test_metrics[f'test_{metric}'].append(test_metrics_epoch[metric])
                
                end_time = time.time()

                if scheduler is not None:
                    scheduler.step()
                
                if early_stopping:
                    _val_monitor = val_loss_epoch if early_stopping_monitor == 'loss' else val_metrics_epoch[early_stopping_monitor]
                    
                    if self.mode_dict[early_stopping_mode](_val_monitor, best_val_monitor):
                        best_val_monitor = _val_monitor
                        best_epoch = epoch
                        best_model = self.model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self.model.load_state_dict(best_model)
                            early_stopping_reached = True

                print_table_row()
        except KeyboardInterrupt:
            print(f"|{'- Keyboard Interrupt -'.center(header_len-2)}|")

        print("-"*header_len)
        print()

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "learning_rate": learning_rates,
            **train_metrics,
            **val_metrics,
            **test_metrics
        }

    def predict(self, data : DataLoader | torch.Tensor):
        """
        Generates predictions for the given data.

        Args:
            data (DataLoader or dict or torch.Tensor): 
                - If DataLoader, generates predictions for the entire dataset.
                - If Tensor, generates a prediction for the single input.

        Returns:
            np.ndarray: Array of predictions.
        """
        self.model.eval()
        predictions = []

        with torch.inference_mode():
            if isinstance(data, torch.utils.data.DataLoader):
                for batch in tqdm(data, desc='Prediction'):
                    input_tensor = batch['input'].to(self.device)
                    output = self.model(input_tensor)
                    pred = self._output_parse(output)
                    predictions.append(pred.cpu().numpy())
                return np.concatenate(predictions, axis=0)
            
            # Single input case (dict or Tensor)
            elif isinstance(data, torch.Tensor):
                input_tensor = data.to(self.device)
                output = self.model(input_tensor).unsqueeze(0)
                pred = self._output_parse(output)
                return pred.cpu().numpy()
            else:
                raise TypeError("Input must be a DataLoader, or torch.Tensor")
    
    def test(self, data_loader, criterion, metrics: dict = {}):
        """
        Evaluates the model on the given data loader.

        Args:
            data_loader (DataLoader): DataLoader for the data to evaluate.
            criterion (torch.nn.Module): Loss function.
            metrics (dict, optional): Dictionary of metrics to compute. Defaults to {}.
            task (str, optional): Task description for progress bar. Defaults to 'Testing'.

        Returns:
            (loss, metrics): Tuple containing the loss and metrics.
        """
        return self._validate_epoch(data_loader, criterion, metrics, task='Testing')

    def _train_epoch(self, train_loader, optimizer, criterion, metrics):
        train_loss = 0
        train_metrics = {metric: 0 for metric in metrics.keys()}

        self.model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', leave=False)
        for batch_idx, batch in progress_bar:
            input = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            optimizer.zero_grad()
            output = self.model(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = self._output_parse(output)
            for metric in metrics.keys():
                train_metrics[metric] += metrics[metric](pred, target)

            progress_bar.set_postfix({'loss': train_loss / (batch_idx + 1)})

        return train_loss / len(train_loader), {metric: train_metrics[metric] / len(train_loader) for metric in train_metrics}

    def _validate_epoch(self, val_loader, criterion, metrics, task='Evaluation'):
        val_loss = 0
        val_metrics = {metric: 0 for metric in metrics.keys()}

        self.model.eval()
        with torch.inference_mode():
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=task, leave=False)
            for batch_idx, batch in progress_bar:
                input = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                
                output = self.model(input)
                loss = criterion(output, target)

                val_loss += loss.item()

                pred = self._output_parse(output)
                for metric in metrics.keys():
                    val_metrics[metric] += metrics[metric](pred, target)
                progress_bar.set_postfix({'loss': val_loss / (batch_idx + 1)})

        return val_loss / len(val_loader), {metric: val_metrics[metric] / len(val_loader) for metric in val_metrics}
    
    def _output_parse(self, output): # modify based on desired output
        return torch.max(output, 1)[1]
