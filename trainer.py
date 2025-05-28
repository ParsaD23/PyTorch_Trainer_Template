import torch
from tqdm import tqdm
import numpy as np
import time
import operator

class Trainer:
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

        def print_table_header():
            base = f"| {'Epoch':<6}| {'lr':<10}| {'train_loss':<11}|"
            val_loss_str = f" {'val_loss':<11}|" if val_loader is not None and val_loss is not None else ""

            monitor_str = ""
            if early_stopping and early_stopping_monitor != 'loss':
                monitor_str = f" {f'train_{early_stopping_monitor}':<20}| {f'val_{early_stopping_monitor}':<20}|"

            time_str = f" {'time (s)':<10}|"

            header = base + val_loss_str + monitor_str + time_str
            header_len = len(header)

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
                # monitor_str = f" train_{early_stopping_monitor}: {train_val:.4f} | val_{early_stopping_monitor}: {val_val:.4f} |"
            
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

        for epoch in range(max_epochs):
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
                        break

            print_table_row()
        print_table_row()

        print("-"*header_len)

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "learning_rate": learning_rates,
            **train_metrics,
            **val_metrics,
            **test_metrics
        }

    def predict(self, data_loader):
        predictions = []

        self.model.eval()
        with torch.inference_mode():
            for batch in tqdm(data_loader, desc='Prediction'):
                input = batch['input'].to(self.device)
                output = self.model(input)

                pred = self._output_parse(output)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)
    
    def test(self, data_loader, criterion, metrics: dict = {}, task='Testing'):
        return self._validate_epoch(data_loader, criterion, metrics, task=task)

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