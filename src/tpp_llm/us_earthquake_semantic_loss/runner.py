"""
TPP-LLM Runner (Updated with Gradient Accumulation)
"""
from typing import Dict, Tuple, List, Union

import numpy as np
import torch
import transformers
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json
import os

from src.tpp_llm.us_earthquake_semantic_loss.model import TPPLLMModel
from src.tpp_llm.us_earthquake_semantic_loss.analysis import run_full_evaluation,visualize_value_consistency

class TPPLLMRunner(object):
    """
    TPP-LLM Runner
    """

    def __init__(
        self, model: TPPLLMModel, beta_type: float = 1, beta_time: float = 1, device: Union[str, torch.device] = 'cpu'
        ,weight_path:str='',load_flag:bool=False):
        """
        Initialize the TPP-LLM runner
        """
        self.model = model
        self.beta_type = beta_type
        self.beta_time = beta_time
        self.device = torch.device(device)

        self.num_train_epochs = 0
        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.global_step = 0
        self.scheduler = None
        self.optimizer = None

        if(load_flag):
            self.load(weight_path)
        self.move_to_numpy_prev = lambda list_tensors: [tensor[1:].numpy(force=True) for tensor in list_tensors]
        self.move_to_numpy_next = lambda list_tensors: [tensor[:-1].numpy(force=True) for tensor in list_tensors]

    def run_batch(self, batch: Dict[str, list], phase: str) \
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray],List[np.ndarray]]:
        """
        Run a batch of event sequences
        """
        import pprint
        # ================== デバッグコード START ==================
        if not hasattr(self, 'has_printed_batch'):
            output_filename = 'batch_content.txt'
            print(f"--- 最初のバッチの1〜2個目のシーケンスを {output_filename} に保存します ---")

            with open(output_filename, 'w', encoding='utf-8') as f:
                first_sequence = {key: value[0] for key, value in batch.items()}
                f.write("--- 1st Sequence in Batch ---\n")
                pprint.pprint(first_sequence, stream=f)
                f.write("\n" + "="*50 + "\n\n")

                if len(batch['type_text']) > 1:
                    second_sequence = {key: value[1] for key, value in batch.items()}
                    f.write("--- 2nd Sequence in Batch ---\n")
                    pprint.pprint(second_sequence, stream=f)
                else:
                    f.write("--- 2nd Sequence in Batch ---\n")
                    f.write("バッチサイズが1のため、2個目のシーケンスはありません。\n")

            print("--- 保存が完了しました ---")
            self.has_printed_batch = True
        # =================== デバッグコード END ===================

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

            # Get the loss terms
            batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses, loss_concept= self.model.compute_loss(batch)
            loss_tpp = torch.sum(
                batch_nll_losses + self.beta_type * batch_type_losses + self.beta_time * batch_time_losses)
            
            # 全てのLossを合計
            batch_loss = loss_tpp + loss_concept 

            # Print metrics
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = - batch_nll_losses.numpy(force=True)
            
            val_loss_cnc = loss_concept.float().cpu().item()
            

            metrics = {
                'batch_loss': batch_loss.float().cpu().item(),
                'loss_tpp': loss_tpp.float().cpu().item(),
                'loss_cnc': loss_concept.float().cpu().item(),
                'batch_event_nums': batch_event_nums.sum().item(),
                'batch_log_likelihood': batch_log_likelihoods.sum().item() / batch_event_nums.sum().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.global_step / self.num_training_steps * self.num_train_epochs,
            }
            print(metrics)

            # ★ 修正: ここでの backward や optimizer.step は削除し、呼び出し元の run_epoch に委ねます
            # （ここで return して、勾配蓄積は外側のループで制御します）
            
            # ただし、batch_loss 自体は backward するために Tensor のまま返す必要があります
            # 呼び出し元で `batch_loss.backward()` するため、ここでは値を返すだけにします
            
            # ★ 戻り値に batch_loss (Tensor) を追加します (10個目の戻り値)
            return batch_event_nums, batch_log_likelihoods, [], [], [], [], [], val_loss_cnc,batch_loss,[]

        elif phase == 'eval':
            self.model.eval()

            batch_event_times = batch['time_since_start']
            batch_event_types = batch['type_event']
            batch_event_time_deltas = batch['time_since_last_event']
# ★追加: マグニチュードも取得
            batch_event_mags = batch['magnitude']

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
            # ★追加: マグニチュードもShiftしてNumpy化 (Ground Truth用)
            batch_event_mags_shifted = self.move_to_numpy_prev(batch_event_mags)

            # Eval時は batch_loss は計算しないので None を返す
            return batch_event_nums, batch_log_likelihoods, batch_event_types_shifted, batch_next_event_types_shifted, \
                   batch_event_times_shifted, batch_next_event_times_shifted,batch_event_time_deltas_shifted,0.0, None,batch_event_mags_shifted

        else:
            raise KeyError(f'Unknown phase: {phase}.')

    def run_epoch(self, dataloader: DataLoader, phase: str, result_save_path:str, gradient_accumulation_steps: int = 1) -> dict:
        """
        Run an epoch (Updated for Gradient Accumulation)
        :param gradient_accumulation_steps: 勾配蓄積数 (デフォルト1)
        """

        total_log_likelihood = 0
        total_num_events = 0
        total_loss_cnc = 0.0
        total_loss_dst = 0.0
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
        metrics = {}
        all_event_time_deltas = []
        
        raw_event_data_list = []

        # ★ Trainフェーズ開始時に勾配を初期化
        if phase == 'train':
            self.optimizer.zero_grad()

        # ★ enumerate でステップ数を取得
        for step, batch in enumerate(tqdm(dataloader)):
            
            # run_batch から batch_loss (Tensor) も受け取るように変更
            # 戻り値のアンパック: 最後に追加した batch_loss を受け取る
            event_nums, log_likelihoods, event_types, event_type_preds, event_times, event_time_preds, event_time_deltas, b_loss_cnc, batch_loss_tensor,event_mags = \
                self.run_batch(batch=batch, phase=phase)
            
            total_log_likelihood += np.sum(log_likelihoods)
            total_num_events += np.sum(event_nums)
            
            if phase == 'train':
                total_loss_cnc += b_loss_cnc
                num_batches += 1

                # ========================================================
                # ★★★ Gradient Accumulation Implementation ★★★
                # ========================================================
                
                # 1. 損失を蓄積数で割る (平均化のため)
                loss = batch_loss_tensor / gradient_accumulation_steps
                
                # 2. 勾配計算 (蓄積される)
                loss.backward()
                
                # 3. 指定ステップ数ごとに更新を実行 (またはデータローダーの最後)
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    
                    # 勾配クリッピング (必要なら有効化)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # デバッグ用: 埋め込み層の勾配確認 (既存コード)
                    '''
                    emb = self.model.llm.get_input_embeddings()
                    if emb.weight.grad is not None:
                        emb.weight.grad[:self.model.old_vocab_size].zero_()
                    '''

                    # 1. 既存トークンの勾配をゼロにする (固定処理)
                    emb = self.model.llm.get_input_embeddings()
                    if emb.weight.grad is not None:
                        emb.weight.grad[:self.model.old_vocab_size].zero_()

                    if emb.weight.grad is not None:
                        nonzero = (emb.weight.grad.abs().sum(dim=1) > 0).sum().item()
                        print(f"[grad-check] nonzero embedding rows: {nonzero} (should equal num_added_tokens)")
            

                    # --------------------------------------------------------
                    # ★★★ 2. ここで確認する (更新直前・消去前) ★★★
                    # --------------------------------------------------------
                    if emb.weight.grad is None:
                        print("⚠️ 警告: emb.weight.grad が None です (更新されていません)")
                    else:
                        # --- A. 既存トークン (固定したい部分) ---
                        # 0番目 〜 old_vocab_size まで
                        original_grad_norm = emb.weight.grad[:self.model.old_vocab_size].norm().item()
                        print(f"既存トークンの勾配ノルム: {original_grad_norm:.6f}")

                        # --- B. 追加トークン (学習したい部分) ---
                        # old_vocab_size 〜 最後 まで
                        added_grad_norm = emb.weight.grad[self.model.old_vocab_size:].norm().item()
                        print(f"追加トークンの勾配ノルム: {added_grad_norm:.6f}")
                    # --------------------------------------------------------
                    
                    # パラメータ更新
                    self.optimizer.step()
                    
                    # 学習率更新
                    self.scheduler.step()
                    
                    # 勾配初期化 (次の蓄積サイクルのため)
                    self.optimizer.zero_grad()
                    
                    # Global Step 更新
                    self.global_step += 1

                    
            
                    

                # ========================================================

            if phase == 'eval':
                for i in range(len(event_types)):
                    raw_event_data_list.append({
                        'true_types': event_types[i].tolist() if isinstance(event_types[i], np.ndarray) else event_types[i],
                        'pred_types': event_type_preds[i].tolist() if isinstance(event_type_preds[i], np.ndarray) else event_type_preds[i],
                        'true_times': event_times[i].tolist() if isinstance(event_times[i], np.ndarray) else event_times[i],
                        'pred_times': event_time_preds[i].tolist() if isinstance(event_time_preds[i], np.ndarray) else event_time_preds[i],
                        # ★追加: ここで保存！
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

        avg_log_likelihood = total_log_likelihood / total_num_events
        metrics['log_likelihood'] = float(avg_log_likelihood)
        metrics['num_events'] = int(total_num_events)

        if phase == 'train' and num_batches > 0:
            metrics['loss_cnc'] = float(total_loss_cnc / num_batches)
            metrics['loss_dst'] = float(total_loss_dst / num_batches)

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
        # ★引数追加
        gradient_accumulation_steps: int = 1
    ) -> None:
        """
        Run the training (Updated)
        """
        # Calculate steps considering accumulation
        self.num_train_epochs = num_train_epochs
        # 学習ステップ数は (データ数 / バッチサイズ / 蓄積数) * エポック数 になる
        steps_per_epoch = len(dataloader_train) // gradient_accumulation_steps
        self.num_training_steps = self.num_train_epochs * steps_per_epoch
        self.num_warmup_steps = int(warmup_ratio * self.num_training_steps)
        self.global_step = 0
        
        self.optimizer = Adam(
            list(self.model.parameters()) ,
            lr=learning_rate
        )

        train_log, train_ac, train_rmse = [], [], []
        val_log, val_ac, val_rmse = [], [], []

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
                result_save_path, "initial_val", seq_scores_val
            )
            
            save_target_dir = result_save_path
            if hasattr(self.model, 'tokenizer'):
                run_full_evaluation(model=self.model, 
                    tokenizer=self.model.tokenizer, 
                    save_dir=save_target_dir, 
                    epoch="initial", 
                    device=self.device
                )
            # ==========================================
            # ★ ここに追加: 学習済みモデルでの初期可視化
            # ==========================================
            try:
                print("Generating initial consistency plots (Pre-trained)...")
                
                # 1. 検証用データローダーから1バッチ取得
                sample_batch = next(iter(dataloader_val))
                
                # 2. マグニチュードの処理
                # リスト形式のバッチをTensorに結合し、GPUへ転送
                raw_mag = torch.cat(sample_batch['magnitude']).to(self.device)
                input_mag = raw_mag.unsqueeze(-1) # (N, 1)

                # 3. 深さの処理
                raw_dep = torch.cat(sample_batch['depth']).to(self.device)
                input_dep = raw_dep.unsqueeze(-1) # (N, 1)

                # 3. 深さの処理
                raw_time = torch.cat(sample_batch['time_since_start']).to(self.device)
                input_time = raw_time.unsqueeze(-1) # (N, 1)

                # 4. モデルのMLPを使って埋め込みベクトルを取得
                with torch.no_grad():
                    embs_mag = self.model.mag_mlp(input_mag)
                    embs_dep = self.model.dep_mlp(input_dep)
                    embs_time = self.model.time_mlp(input_time)

                # 5. 可視化実行 (ファイル名を initial に変更)
                # Magnitude
                plot_path_mag = os.path.join(result_save_path, "mag_consistency_initial.png")
                visualize_value_consistency(
                    input_mag, 
                    embs_mag, 
                    label_name="Magnitude", 
                    save_path=plot_path_mag
                )
                
                # Depth
                plot_path_dep = os.path.join(result_save_path, "depth_consistency_initial.png")
                visualize_value_consistency(
                    input_dep, 
                    embs_dep, 
                    label_name="Depth", 
                    save_path=plot_path_dep
                )
                # Depth
                plot_path_time = os.path.join(result_save_path, "time_consistency_initial.png")
                visualize_value_consistency(
                    input_time, 
                    embs_time, 
                    label_name="Time", 
                    save_path=plot_path_time
                )
                
                print(f"Saved initial consistency plots to {result_save_path}")

            except Exception as e:
                print(f"Initial visualization failed: {e}")
                import traceback
                traceback.print_exc()
            # ==========================================
            

        if dataloader_test:
            metrics_test, test_eval_data = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
            print(f'test metrics: {metrics_test}')
            with open(result_save_path+'/test.txt', 'a') as f:
                f.write(f'test metrics: {metrics_test}\n')
            (all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test, seq_scores_test) = test_eval_data
            quantitative_analysis_func(
                all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test,
                result_save_path, "initial_test", seq_scores_test
            )
            with open(os.path.join(result_save_path, "seq_scores_initial_test.json"), "w") as f:
                json.dump(seq_scores_test, f)

        patience = 10
        epochs_no_improve = 0
        metrics_val_best = None

        # Start training loop
        for epoch in range(num_train_epochs):
            if save_flag or train_flag:
                print(f'epoch: {epoch}')
            
            if dataloader_train and (save_flag or train_flag):
                # ★ 修正: gradient_accumulation_steps を渡す
                metrics_train = self.run_epoch(dataloader_train, phase='train', result_save_path=result_save_path, 
                                               gradient_accumulation_steps=gradient_accumulation_steps)
                if(epoch==0):
                    import subprocess
                    with open(result_save_path+'/val.txt', 'a') as f:
                        subprocess.run(["nvidia-smi"], stdout=f, text=True)
                
                print(f'train metrics of epoch {epoch}: {metrics_train}')
                train_log.append(metrics_train['log_likelihood'])
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
                            # Best Model Visual Analysis
                            save_target_dir = result_save_path
                            if hasattr(self.model, 'tokenizer'):
                                run_full_evaluation(
                                    model=self.model, 
                                    tokenizer=self.model.tokenizer, 
                                    save_dir=save_target_dir, 
                                    epoch=epoch, 
                                    device=self.device
                                )
                                # ==========================================
                                # ★ 数値埋め込みの整合性チェック (可視化)
                                # ==========================================
                            try:
                                print("Generating consistency plots...")
                            
                                # 1. 検証データから1バッチだけ取得
                                # バッチはリスト形式なので、next(iter(...)) で辞書を取得
                                sample_batch = next(iter(dataloader_val))
                            
                            # 2. データの整形 (List[Tensor] -> Tensor(N, 1))
                            # モデルのMLPは (Batch*Seq, 1) の形を期待しているので結合します
                            
                            # --- マグニチュード ---
                             # sample_batch['magnitude'] はテンソルのリストなので結合
                                raw_mag = torch.cat(sample_batch['magnitude']).to(self.device)
                                # MLPに入力するため (N, 1) に変形
                                input_mag = raw_mag.unsqueeze(-1)
                            
                            # --- 深さ ---
                                raw_dep = torch.cat(sample_batch['depth']).to(self.device)
                                input_dep = raw_dep.unsqueeze(-1)

                                raw_time = torch.cat(sample_batch['time_since_start']).to(self.device)
                                input_time = raw_time.unsqueeze(-1)
                            
                                # 3. 埋め込みベクトルの取得
                                # モデル内のMLPを直接呼び出してベクトル化します
                                with torch.no_grad():
                                # TPPLLMModelで定義されている mag_mlp / dep_mlp を使用
                                    embs_mag = self.model.mag_mlp(input_mag)
                                    embs_dep = self.model.dep_mlp(input_dep)
                                    embs_time = self.model.time_mlp(input_time)

                            # 4. 可視化関数の実行 (可視化関数はファイルの先頭等に定義済みとする)
                            
                            # Magnitudeのプロット
                                plot_path_mag = os.path.join(result_save_path, f"mag_consistency.png")
                                visualize_value_consistency(
                                    input_mag,      # 数値
                                    embs_mag,       # ベクトル
                                    label_name="Magnitude", 
                                    save_path=plot_path_mag
                                )
                            
                            # Depthのプロット (ここが平坦なら「深さは無視されている」説が立証される)
                                plot_path_dep = os.path.join(result_save_path, f"depth_consistency.png")
                                visualize_value_consistency(
                                    input_dep,      # 数値
                                    embs_dep,       # ベクトル
                                    label_name="Depth", 
                                    save_path=plot_path_dep
                                )

                                plot_path_time = os.path.join(result_save_path, f"time_consistency.png")
                                visualize_value_consistency(
                                    input_time,      # 数値
                                    embs_time,       # ベクトル
                                    label_name="Time", 
                                    save_path=plot_path_time
                                )
                            
                                print(f"Saved consistency plots to {result_save_path}")

                            except Exception as e:
                                print(f"Visualization failed: {e}")
                                import traceback
                                traceback.print_exc()
                        # ==========================================
                            

                        
                        (all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true, seq_scores_val) = eval_data
                        quantitative_analysis_func(
                            all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true,
                            result_save_path, "val", seq_scores_val
                        )
                    else:
                        epochs_no_improve += 1
                        print(f'No improvement for {epochs_no_improve} epochs')
                        f.write(f'No improvement for {epochs_no_improve} epochs\n')

                val_log.append(metrics_val['log_likelihood'])
                val_ac.append(metrics_val['accuracy'])
                val_rmse.append(metrics_val['rmse'])

                if epochs_no_improve >= patience:
                    with open(result_save_path+'/val.txt', 'a') as f:
                        f.write(f"Early stopping triggered at epoch {epoch}")
                    break

            if dataloader_test and (save_flag or train_flag):
                metrics_test, test_eval_data = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
                print(f'test metrics of epoch {epoch}: {metrics_test}')
                with open(result_save_path+'/test.txt', 'a') as f:
                    f.write(f'test metrics of epoch {epoch}: {metrics_test}\n')
                
                if(epochs_no_improve == 0):
                    (all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test, seq_scores_test) = test_eval_data
                    quantitative_analysis_func(
                        all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test,
                        result_save_path, f"test", seq_scores_test
                    )
                    with open(os.path.join(result_save_path, "seq_scores_best_test.json"), "w") as f:
                        json.dump(seq_scores_test, f)

            if(save_flag or train_flag):
                import matplotlib.pyplot as plt
                epochs_range = range(0, len(train_log))
                
                plt.figure(figsize=(8, 5))
                plt.plot(epochs_range, train_log, marker='o', linestyle='-', color='blue', label='Training LL')
                plt.plot(epochs_range, val_log, marker='o', linestyle='-', color='orange', label='Validation LL')
                plt.xlabel('Epoch')
                plt.ylabel('Log-Likelihood')
                plt.title('Log-Likelihood')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(figure_path+"/logl.png")
                plt.close()

    def save(self, model_weight_path: str, weight_path: str):
        self.model.save_models(model_weight_path)
        filtered_weights = {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith("model.") and not k.startswith("lm_head")
        }
        torch.save(filtered_weights, weight_path+'/model.pth')

    def load(self, weight_path: str):
        self.model.load_state_dict(torch.load(weight_path+'/model.pth'), strict=False)