import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import time
import operator
import random

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
            freeze_epochs=None,
            metrics: dict = {},
            fast_dev_run=False,
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
            freeze_epochs (int, optional): Number of epochs to freeze the model. Defaults to None.
            metrics (dict, optional): Dictionary of metrics to compute. Defaults to {}.
            fast_dev_run(bool, optional): Train the model on a single batch to check model soundness. Defaults to False.

        Returns:
            dict: Dictionary containing training, validation, and test losses and metrics.
        """
        def print_table_header():
            base = f"| {'Epoch':<6}| {'lr':<10}| {'train_loss':<11}|"
            val_loss_str = f" {'val_loss':<11}|" if val_loader is not None and val_metrics['loss'] is not None else ""

            monitor_str = ""
            if early_stopping and early_stopping_monitor != 'loss':
                monitor_str = f" {f'train_{early_stopping_monitor}':<20}| {f'val_{early_stopping_monitor}':<20}|"

            time_str = f" {'time':<10}|"

            header = base + val_loss_str + monitor_str + time_str
            header_len = len(header)
            
            print()
            print("-"*header_len)
            print(header)
            print("-"*header_len)

            return header_len

        def print_table_row():
            elapsed_time = end_time - start_time

            base_row = f"| {epoch:<6}| {lr:<10.3E}| {train_metrics_epoch['loss']:<11.3E}|"
            
            val_loss_str = f" {val_metrics_epoch['loss']:<11.3E}|" if val_loader is not None else ""
            
            monitor_str = ""
            if early_stopping and early_stopping_monitor != 'loss':
                train_val = train_metrics_epoch[early_stopping_monitor]
                val_val = val_metrics_epoch[early_stopping_monitor]
                monitor_str = f" {train_val:<20.4f}| {val_val:<20.4f}|"
            
            time_str = f" {format_duration(elapsed_time):<10}|"
            early_stop_flag = " (!) Early stopping" if early_stopping_reached else ""
            
            print(base_row + val_loss_str + monitor_str + time_str + early_stop_flag)

        train_metrics = {'loss': []}
        val_metrics = {'loss': []}
        test_metrics = {'loss': []}

        train_metrics.update({f'train_{metric}': [] for metric in metrics.keys()})
        val_metrics.update({f'val_{metric}': [] for metric in metrics.keys()})
        test_metrics.update({f'test_{metric}': [] for metric in metrics.keys()})

        learning_rates = []

        best_val_monitor = float('inf') if early_stopping_mode == 'min' else float('-inf')
        best_model = None
        patience_counter = 0
        early_stopping_reached = False

        header_len = print_table_header()
        
        if fast_dev_run:
            subset_indices = random.sample(range(len(train_loader.dataset)), train_loader.batch_size)
            train_loader = DataLoader(Subset(train_loader.dataset, subset_indices), batch_size=train_loader.batch_size)

            if val_loader:
                subset_indices = random.sample(range(len(val_loader.dataset)), val_loader.batch_size)
                val_loader = DataLoader(Subset(val_loader.dataset, subset_indices), batch_size=val_loader.batch_size)

            if test_loader:
                subset_indices = random.sample(range(len(train_loader.dataset)), test_loader.batch_size)
                train_loader = DataLoader(Subset(train_loader.dataset, subset_indices), batch_size=test_loader.batch_size)

        try:
            for epoch in range(max_epochs):
                if early_stopping_reached:
                    break

                start_time = time.time()
                
                if freeze_epochs is not None:
                    if epoch < freeze_epochs:
                        self.model.freeze()
                    else:
                        self.model.unfreeze()

                lr = optimizer.param_groups[0]['lr']
                learning_rates.append(lr)

                train_metrics_epoch = self._train_epoch(train_loader, optimizer, criterion, metrics)
                train_metrics['loss'].append(train_metrics_epoch['loss'])
                
                if val_loader is not None:
                    val_metrics_epoch = self._validate_epoch(val_loader, criterion, metrics, task='Evaluation')
                    train_metrics['loss'].append(val_metrics_epoch['loss'])
                
                if test_loader is not None:
                    test_metrics_epoch = self._validate_epoch(test_loader, criterion, metrics, task='Test Evaluation')
                    test_metrics['loss'].append(test_metrics_epoch['loss'])

                for metric in metrics.keys():
                    train_metrics[f'train_{metric}'].append(train_metrics_epoch[metric])
                    if val_loader is not None:
                        val_metrics[f'val_{metric}'].append(val_metrics_epoch[metric])
                    if test_loader is not None:
                        test_metrics[f'test_{metric}'].append(test_metrics_epoch[metric])
                
                end_time = time.time()

                if scheduler is not None:
                    scheduler.step()
                
                if early_stopping and not fast_dev_run:
                    _val_monitor = val_metrics_epoch[early_stopping_monitor]
                    
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
            **train_metrics,
            **val_metrics,
            **test_metrics,
            "learning_rate": learning_rates,
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
                    pred = self.model.output_parse(output)
                    predictions.append(pred.cpu().numpy())
                return np.concatenate(predictions, axis=0)
            
            # Single input case (dict or Tensor)
            elif isinstance(data, torch.Tensor):
                input_tensor = data.to(self.device)
                output = self.model(input_tensor).unsqueeze(0)
                pred = self.model.output_parse(output)
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

        num_samples = 0
        all_preds = []
        all_targets = []

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
            
            batch_size = input.size(0)
            train_loss += loss.item()
            num_samples += batch_size

            pred = self.model.output_parse(output)
            
            all_preds.append(pred)
            all_targets.append(target)

            progress_bar.set_postfix({'loss': train_loss / num_samples})

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        results = {'loss': train_loss / num_samples}
        results.update({metric: metrics[metric](all_preds, all_targets) for metric in metrics})
        
        return results

    def _validate_epoch(self, val_loader, criterion, metrics, task='Evaluation'):
        val_loss = 0
        
        num_samples = 0
        all_preds = []
        all_targets = []

        self.model.eval()
        with torch.inference_mode():
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=task, leave=False)
            for batch_idx, batch in progress_bar:
                input = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                
                output = self.model(input)
                loss = criterion(output, target)

                batch_size = input.size(0)
                val_loss += loss.item() * batch_size
                num_samples += batch_size

                pred = self.model.output_parse(output)

                all_preds.append(pred)
                all_targets.append(target)
                
                progress_bar.set_postfix({'loss': val_loss / num_samples})

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        results = {'loss': val_loss / num_samples}
        results.update({metric: metrics[metric](all_preds, all_targets) for metric in metrics})
        
        return results

# ------------ util functions ------------

def format_duration(seconds):
    secs = int(seconds)
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    secs = secs % 60
    ms = int(seconds * 1000)

    if hours > 0:
        return f"{hours} h {minutes} m"
    elif minutes > 0:
        return f"{minutes} m {secs} s"
    elif secs > 0:
        return f"{secs} s {ms} ms"
    else:
        return f"{ms} ms"
