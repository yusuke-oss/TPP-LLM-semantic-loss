"""
TPP-LLM Runner
Handles the training loop, evaluation, and gradient accumulation.
"""
import os
import json
from typing import Dict, Tuple, List, Union

import numpy as np
import torch
import transformers
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.tpp_llm.us_earthquake_semantic_loss.model import TPPLLMModel

class TPPLLMRunner(object):
    """
    TPP-LLM Runner
    """

    def __init__(
        self, model: TPPLLMModel, beta_type: float = 1, beta_time: float = 1,beta_semantic: float = 1.0, device: Union[str, torch.device] = 'cpu'
        ,weight_path:str='',load_flag:bool=False):
        """
        Initialize the TPP-LLM runner.

        :param model: The TPPLLMModel instance.
        :param beta_type: Loss coefficient for event type prediction.
        :param beta_time: Loss coefficient for event time prediction.
        :param beta_semantic: Loss coefficient for semantic alignment.
        :param device: Execution device.
        """


        self.model = model
        self.beta_type = beta_type
        self.beta_time = beta_time
        self.beta_semantic = beta_semantic
        self.device = torch.device(device)

        self.num_train_epochs = 0
        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.global_step = 0
        self.scheduler = None
        self.optimizer = None

        if(load_flag):
            self.load(weight_path)

        # Lambda functions to detach tensors and move them to CPU as numpy arrays
        self.move_to_numpy_prev = lambda list_tensors: [tensor[1:].numpy(force=True) for tensor in list_tensors]
        self.move_to_numpy_next = lambda list_tensors: [tensor[:-1].numpy(force=True) for tensor in list_tensors]

    def run_batch(self, batch: Dict[str, list], phase: str) \
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray],List[np.ndarray]]:
        """
        Run a single batch of event sequences.

        Returns:
            Tuple containing:
            - event_nums: Number of events per sequence
            - log_likelihoods: Log-likelihoods of the sequences
            - event_types_shifted: True event types (shifted for prediction)
            - next_event_types_shifted: Predicted next event types
            - event_times_shifted: True event times (shifted)
            - next_event_times_shifted: Predicted next event times
            - event_time_deltas_shifted: True time deltas (shifted)
            - semantic_loss: Semantic alignment loss (float)
            - batch_loss: Total combined loss (Tensor, for backward pass)
            - event_mags_shifted: True event magnitudes (shifted)
        """

        # Move inputs to the target device
        batch = {
            'time_since_start': [_seq.to(self.device) for _seq in batch['time_since_start']],
            'time_since_last_event': [_seq.to(self.device) for _seq in batch['time_since_last_event']],
            'type_event': [_seq.to(self.device) for _seq in batch['type_event']],
            'type_text':batch['type_text'],
            'magnitude':[_seq.to(self.device) for _seq in batch['magnitude']],
            'depth':[_seq.to(self.device) for _seq in batch['depth']],
        }

        if phase == 'train':
            self.model.train()

            # Forward pass and compute loss
            # (model.compute_loss returns: event_nums, NLL_losses, Type_losses, Time_losses, raw_semantic_loss)
            batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses, raw_semantic_loss = self.model.compute_loss(batch)
            
            # TPP Base Loss (NLL + Type + Time)
            loss_tpp = torch.sum(
                batch_nll_losses + self.beta_type * batch_type_losses + self.beta_time * batch_time_losses)
            
            # Total Loss 
            weighted_semantic_loss = self.beta_semantic * raw_semantic_loss
            batch_loss = loss_tpp + weighted_semantic_loss

            # Print metrics
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = - batch_nll_losses.numpy(force=True)
            val_semantic_loss = weighted_semantic_loss.float().cpu().item()
            

            metrics = {
                'batch_loss': batch_loss.float().cpu().item(),
                'loss_tpp': loss_tpp.float().cpu().item(),
                'semantic_loss': val_semantic_loss,
                'batch_event_nums': batch_event_nums.sum().item(),
                'batch_log_likelihood': batch_log_likelihoods.sum().item() / batch_event_nums.sum().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.global_step / self.num_training_steps * self.num_train_epochs,
            }
            print(metrics)

            return batch_event_nums, batch_log_likelihoods, [], [], [], [], [], \
            val_semantic_loss,batch_loss,[]

        elif phase == 'eval':
            self.model.eval()

            batch_event_times = batch['time_since_start']
            batch_event_types = batch['type_event']
            batch_event_time_deltas = batch['time_since_last_event']
            batch_event_mags = batch['magnitude']

            # Predict next events
            batch_event_nums, batch_log_likelihoods, batch_next_event_types, batch_next_event_times = \
                self.model.predict_next_events(batch)

            # Shift and convert tensors
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = batch_log_likelihoods.numpy(force=True)
            batch_event_types_shifted = self.move_to_numpy_prev(batch_event_types)
            batch_event_times_shifted = self.move_to_numpy_prev(batch_event_times)
            batch_next_event_types_shifted = self.move_to_numpy_next(batch_next_event_types)
            batch_next_event_times_shifted = self.move_to_numpy_next(batch_next_event_times)
            batch_event_time_deltas_shifted = self.move_to_numpy_prev(batch_event_time_deltas)
            batch_event_mags_shifted = self.move_to_numpy_prev(batch_event_mags)

            # Return 0.0 and None for loss values during evaluation
            return batch_event_nums, batch_log_likelihoods, batch_event_types_shifted, \
                batch_next_event_types_shifted, batch_event_times_shifted, \
                batch_next_event_times_shifted,batch_event_time_deltas_shifted,0.0, \
                None,batch_event_mags_shifted

        else:
            raise KeyError(f'Unknown phase: {phase}.')

    def run_epoch(self, dataloader: DataLoader, phase: str, result_save_path:str, gradient_accumulation_steps: int = 1) -> dict:
        """
        Run an epoch (Updated for Gradient Accumulation)
        :param gradient_accumulation_steps: 勾配蓄積数 (デフォルト1)
        """

        total_log_likelihood = 0
        total_num_events = 0
        total_semantic_loss = 0.0
        num_batches = 0
        
        raw_event_types = []
        raw_event_type_preds = []
        raw_event_times = []
        raw_event_time_preds = []
        raw_event_time_deltas = []
        
        all_event_types = []
        all_event_type_preds = []
        all_event_times = []
        all_event_time_preds = []
        all_event_time_deltas = []
        
        raw_event_data_list = []
        metrics = {}

        # ★ Trainフェーズ開始時に勾配を初期化
        if phase == 'train':
            self.optimizer.zero_grad()

        # ★ enumerate でステップ数を取得
        for step, batch in enumerate(tqdm(dataloader)):
            
            # Unpack 10 returned items from run_batch
            event_nums, log_likelihoods, event_types, event_type_preds, event_times, \
            event_time_preds, event_time_deltas, semantic_loss, batch_loss_tensor,event_mags = \
            self.run_batch(batch=batch, phase=phase)
            
            total_log_likelihood += np.sum(log_likelihoods)
            total_num_events += np.sum(event_nums)
            
            if phase == 'train':
                total_semantic_loss += semantic_loss
                num_batches += 1

                # 1. Average the loss over the accumulation steps
                loss = batch_loss_tensor / gradient_accumulation_steps
                
                # 2. Backward pass (accumulate gradients)
                loss.backward()
                
                # 3. Update parameters if we reached the accumulation step limit or the end of the dataloader
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    
                    # Zero out the gradients of the original LLM embeddings to keep them frozen
                    emb = self.model.llm.get_input_embeddings()
                    if emb.weight.grad is not None:
                        emb.weight.grad[:self.model.old_vocab_size].zero_()

                    if emb.weight.grad is not None:
                        nonzero = (emb.weight.grad.abs().sum(dim=1) > 0).sum().item()
                        print(f"[grad-check] nonzero embedding rows: {nonzero} (should equal num_added_tokens)")
            

                    
                    # Update the parameters
                    self.optimizer.step()
                    
                    # Update the learning rate
                    self.scheduler.step()
                    
                    # Optimize the model parameters
                    self.optimizer.zero_grad()
                    
                    # Update the Global Step
                    self.global_step += 1

                    
            
                    

                # ========================================================

            if phase == 'eval':
                for i in range(len(event_types)):
                    raw_event_data_list.append({
                        'true_types': event_types[i].tolist() if isinstance(event_types[i], np.ndarray) else event_types[i],
                        'pred_types': event_type_preds[i].tolist() if isinstance(event_type_preds[i], np.ndarray) else event_type_preds[i],
                        'true_times': event_times[i].tolist() if isinstance(event_times[i], np.ndarray) else event_times[i],
                        'pred_times': event_time_preds[i].tolist() if isinstance(event_time_preds[i], np.ndarray) else event_time_preds[i],
                        'true_mags': event_mags[i].tolist() if isinstance(event_mags[i], np.ndarray) else event_mags[i]
                    })
                all_event_types.append(np.concatenate(event_types))
                all_event_type_preds.append(np.concatenate(event_type_preds))
                all_event_times.append(np.concatenate(event_times))
                all_event_time_preds.append(np.concatenate(event_time_preds))
                all_event_time_deltas.append(np.concatenate(event_time_deltas))

                raw_event_types.extend(event_types)
                raw_event_type_preds.extend(event_type_preds)
                raw_event_times.extend(event_times)
                raw_event_time_preds.extend(event_time_preds)
                raw_event_time_deltas.extend(event_time_deltas)

        # Calculate final metrics for the epoch
        avg_log_likelihood = total_log_likelihood / total_num_events
        metrics['log_likelihood'] = float(avg_log_likelihood)
        metrics['num_events'] = int(total_num_events)

        if phase == 'train' and num_batches > 0:
            metrics['semantic_loss'] = float(total_semantic_loss / num_batches) 

        if phase == 'eval':
            all_event_types = np.concatenate(all_event_types)
            all_event_type_preds = np.concatenate(all_event_type_preds)
            all_event_times = np.concatenate(all_event_times)
            all_event_time_preds = np.concatenate(all_event_time_preds)
            all_event_time_deltas = np.concatenate(all_event_time_deltas)

            accuracy = np.mean(all_event_types == all_event_type_preds)
            rmse = np.sqrt(np.mean((all_event_times - all_event_time_preds) ** 2))
            metrics['accuracy'] = float(accuracy)
            metrics['rmse'] = float(rmse)
            
            seq_accuracies = []
            seq_rmses = []
            for true_typ, pred_typ, true_tm, pred_tm in zip(
                raw_event_types, raw_event_type_preds, raw_event_times, raw_event_time_preds
            ):
                seq_acc = np.mean(true_typ == pred_typ)
                seq_accuracies.append(float(seq_acc))
                seq_rmse = np.sqrt(np.mean((true_tm - pred_tm) ** 2))
                seq_rmses.append(float(seq_rmse))
            
            seq_scores = {
                "accuracies": seq_accuracies,
                "rmses": seq_rmses,
                "raw_data": raw_event_data_list
            }
            return metrics, (all_event_types, all_event_type_preds,
                             all_event_times, all_event_time_preds,
                             all_event_time_deltas, seq_scores)

        return metrics

    def run(
        self, quantitative_analysis_func: callable,
        dataloader_train: DataLoader = None, dataloader_val: DataLoader = None,
        dataloader_test: DataLoader = None, learning_rate: float = 5e-4, lr_scheduler_type: str = 'constant',
        num_train_epochs: int = 1, warmup_ratio: float = 0, result_save_path: str = '',
        weight_path: str = '', model_weight_path: str = '', figure_path: str = '', save_flag: bool = False, train_flag: bool = False,
        gradient_accumulation_steps: int = 1
    ) -> None:
        """
        Run the training (Updated)
        """
        # Calculate steps considering accumulation
        self.num_train_epochs = num_train_epochs
        steps_per_epoch = len(dataloader_train) // gradient_accumulation_steps
        self.num_training_steps = self.num_train_epochs * steps_per_epoch
        self.num_warmup_steps = int(warmup_ratio * self.num_training_steps)
        self.global_step = 0
        
        self.optimizer = Adam(
            list(self.model.parameters()) ,
            lr=learning_rate
        )

        if lr_scheduler_type == 'constant':
            self.scheduler = transformers.get_constant_schedule(optimizer=self.optimizer)
        elif lr_scheduler_type == 'constant_with_warmup':
            self.scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps)
        elif lr_scheduler_type == 'linear':
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
        elif lr_scheduler_type == 'cosine':
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps, num_cycles=0.5)
        else:
            raise KeyError(f'Unknown learning rate scheduler type: {lr_scheduler_type}')

        metrics_val_best = None

        # Initial validation
        if dataloader_val:
            metrics_val, eval_data = self.run_epoch(dataloader_val, phase='eval', result_save_path=result_save_path)
            print(f'validation metrics: {metrics_val}')
            metrics_val_best = metrics_val
            with open(result_save_path+'/val.txt', 'a') as f:
                f.write(f'validation metrics: {metrics_val}\n')
            
            (all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true, seq_scores_val) = eval_data
            quantitative_analysis_func(
                all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true,
                result_save_path, "initial_val","initial", seq_scores_val
            )
            
        # ---------------------------------------------------------
        # Initial Test
        # ---------------------------------------------------------
        if dataloader_test:
            metrics_test, test_eval_data = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
            print(f'test metrics: {metrics_test}')
            with open(result_save_path+'/test.txt', 'a') as f:
                f.write(f'test metrics: {metrics_test}\n')

            (all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test, seq_scores_test) = test_eval_data
            quantitative_analysis_func(
                all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test,
                result_save_path, "initial_test","initial", seq_scores_test
            )
        
        patience = 10
        epochs_no_improve = 0

        # Start training loop
        for epoch in range(num_train_epochs):
            if save_flag or train_flag:
                print(f'epoch: {epoch}')
            
            if dataloader_train and (save_flag or train_flag):
                metrics_train = self.run_epoch(dataloader_train, phase='train', result_save_path=result_save_path, 
                                               gradient_accumulation_steps=gradient_accumulation_steps)
                if(epoch==0):
                    import subprocess
                    with open(result_save_path+'/val.txt', 'a') as f:
                        subprocess.run(["nvidia-smi"], stdout=f, text=True)
                
                print(f'train metrics of epoch {epoch}: {metrics_train}')
                with open(result_save_path+'/train.txt', 'a') as f:
                    f.write(f'train metrics of epoch {epoch}: {metrics_train}\n')

            if dataloader_val and (save_flag or train_flag):
                metrics_val, eval_data = self.run_epoch(dataloader_val, phase='eval', result_save_path=result_save_path)
                print(f'validation metrics of epoch {epoch}: {metrics_val}')

                with open(result_save_path+'/val.txt', 'a') as f:
                    f.write(f'validation metrics of epoch {epoch}: {metrics_val}\n')

                    if metrics_val_best is None or metrics_val['log_likelihood'] > metrics_val_best['log_likelihood']:
                        metrics_val_best = metrics_val
                        epochs_no_improve = 0
                        print(f'new best validation metrics')
                        f.write(f'new best validation metrics\n')

                        if save_flag:
                            self.save(model_weight_path, weight_path)

                        (all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true, seq_scores_val) = eval_data
                        quantitative_analysis_func(
                            all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true,
                            result_save_path, "val", epoch, seq_scores_val
                        )
                    else:
                        epochs_no_improve += 1
                        print(f'No improvement for {epochs_no_improve} epochs')
                        f.write(f'No improvement for {epochs_no_improve} epochs\n')

                if epochs_no_improve >= patience:
                    with open(result_save_path+'/val.txt', 'a') as f:
                        f.write(f"Early stopping triggered at epoch {epoch}")
                    break

            if dataloader_test and (save_flag or train_flag):
                metrics_test, test_eval_data = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
                print(f'test metrics of epoch {epoch}: {metrics_test}')
                with open(result_save_path+'/test.txt', 'a') as f:
                    f.write(f'test metrics of epoch {epoch}: {metrics_test}\n')
                # Only save detailed test analysis if it's the best epoch so far
                if(epochs_no_improve == 0):
                    (all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test, seq_scores_test) = test_eval_data
                    quantitative_analysis_func(
                        all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test,
                        result_save_path, f"test", epoch, seq_scores_test
                    )
                    with open(os.path.join(result_save_path, "seq_scores_best_test.json"), "w") as f:
                        json.dump(seq_scores_test, f)


    def save(self, model_weight_path: str, weight_path: str):
        self.model.save_models(model_weight_path)
        filtered_weights = {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith("model.") and not k.startswith("lm_head")
        }
        torch.save(filtered_weights, weight_path+'/model.pth')

    def load(self, weight_path: str):
        self.model.load_state_dict(torch.load(weight_path+'/model.pth'), strict=False)