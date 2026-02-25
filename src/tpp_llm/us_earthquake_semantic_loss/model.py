"""
TPP-LLM Model
"""
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
from peft import get_peft_model, PeftConfig,PeftModel 
from torch import Tensor
from torch.nn import functional
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig,AddedToken
import numpy as np
from src.tpp_llm.us_earthquake_semantic_loss.layers import TimePositionalEncoding, TimeShiftedPositionalEncoding,LatLonMLPEncoding,LearnableTemporalEncoding

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F



class TPPLLMModel(nn.Module):
    """
    TPP-LLM Model
    """

    def __init__(
        self, model_name: str, num_event_types: int, num_integral_samples: int, temporal_emb_type: str,
        temporal_emb_first: bool = False, prompt: str = '', bnb_config: BitsAndBytesConfig = None,
        peft_config: PeftConfig = None, device: Union[str, torch.device] = 'cpu',
        model_weight_path:str ='',save_flag:bool=False,load_flag:bool=False,train_flag:bool=False,# ★追加: 事前学習済み重みのパスを受け取る引数
        alpha_concept: float = 1.0,  # 概念アンカーLossの重み

        **kwargs):
        """
        Initialize the TPP-LLM model

        :param model_name: LLM name
        :param num_event_types: number of event types
        :param num_integral_samples: number of samples in the intensity integral
        :param temporal_emb_type: temporal embedding type
        :param temporal_emb_first: temporal embedding first (before the text embedding) for each event
        :param prompt: prompt before the event sequences
        :param bnb_config: bits and bytes configuration
        :param peft_config: PEFT configuration
        :param device: device
        """
        super().__init__()
        self.alpha_concept = alpha_concept
        

        

        
        

        
        
            

        if(save_flag==True or train_flag==True):
            # Load the LLM
            
            self.device = torch.device(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float32,
                device_map=self.device,
            )

            # Apply the PEFT config
            if peft_config is not None:
                self.llm = get_peft_model(self.llm, peft_config)
                self.llm.train()
                self.llm.print_trainable_parameters()
                '''
                f = open('model_or_parameter/model.txt', 'a')
                f.write(str(self.llm.print_trainable_parameters()))
                f.close()
                '''
            else:
                self.llm.eval()
                for param in self.llm.parameters():
                    param.requires_grad = False
            
            self.old_vocab_size = self.llm.get_input_embeddings().num_embeddings
            
            # 1. byte 系トークン（共通で使える）
            byte_tokens1 = [
                AddedToken(f"<|byte_{i}|>", special=True, lstrip=False, rstrip=False)
                for i in range(256)
            ]
            byte_tokens2 = [
                AddedToken(f"<|byte2_{i}|>", special=True, lstrip=False, rstrip=False)
                for i in range(256)
            ]
            

            # 2. sign トークン（0/1）
            sign_tokens = [
                AddedToken(f"<|sign_{i}|>", special=True, lstrip=False, rstrip=False)
                for i in range(2)
            ]
            '''
            # 3. exponent トークン
            # float32 用 (0〜255)
            exp32_tokens = [
                AddedToken(f"<|exp32_{i}|>", special=True, lstrip=False, rstrip=False)
                for i in range(256)
            ]
            '''
            # float16 用 (0〜31)
            exp16_tokens = [
                AddedToken(f"<|exp_{i}|>", special=True, lstrip=False, rstrip=False)
                for i in range(32)
            ]

            

            
            '''
            types_tokens = [
                AddedToken(f"<|type_{i}|>", special=True, lstrip=False, rstrip=False)
                for i in range(0,31)
            ]
            AddedToken("<|first_time|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|ongoing|>", special=True, lstrip=False, rstrip=False),
            # 一括追加（special_tokens=True により added_tokens.json にも反映される）
            self.tokenizer.add_tokens(types_tokens, special_tokens=True)
            '''
            
            # start_of_event / end_of_event トークンも追加する場合
            event_tokens = [
            
                AddedToken("<|start_of_event|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|end_of_event|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|time_prefix|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|type_prefix|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|magnitude_prefix|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|depth_prefix|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|im_start|>", special=True, lstrip=False, rstrip=False),
                AddedToken("<|im_end|>", special=True, lstrip=False, rstrip=False),
                #AddedToken("<|mag_distribution|>", special=True, lstrip=False, rstrip=False),
                #AddedToken("<|mag_params|>", special=True, lstrip=False, rstrip=False),
                #AddedToken("<|depth_distribution|>", special=True, lstrip=False, rstrip=False),
                #AddedToken("<|depth_params|>", special=True, lstrip=False, rstrip=False),
                # 必要なら他のプレフィックス
                #AddedToken("<|earthquake_prefix|>", special=True, lstrip=False, rstrip=False),
            ]
            # まとめて追加
            all_tokens = event_tokens
            #all_tokens = byte_tokens1 + byte_tokens2+ sign_tokens + exp16_tokens + event_tokens
            # 追加前の語彙サイズを保存
            self.old_vocab_size = len(self.tokenizer)
            # トークナイザに追加
            self.tokenizer.add_tokens(all_tokens, special_tokens=True)
            # 埋め込みサイズを拡張（必須！）
            self.llm.resize_token_embeddings(len(self.tokenizer))
            '''
            # 新しいトークンID範囲を取得
            new_token_ids = list(range(self.old_vocab_size, len(self.tokenizer)))

            # 埋め込み層にアクセス
            emb = self.llm.get_input_embeddings()
            '''
            '''
            with torch.no_grad():
                # 1. ゼロ初期化
                # for idx in new_token_ids:
                #     emb.weight[idx].fill_(0.0)

                # 2. 平均初期化
                mean_vec = emb.weight[:self.old_vocab_size].mean(dim=0, keepdim=True)
                emb.weight[new_token_ids] = mean_vec.expand(len(new_token_ids), -1)
            '''
            

            # ← このすぐ後に print を置く
            print(f"[TPPLLMModel] old_vocab_size={getattr(self,'old_vocab_size', None)}, "
                f"total_vocab={len(self.tokenizer)}, "
                f"embedding_dim={self.llm.get_input_embeddings().embedding_dim}")

            # --- 追加: 追加トークンのみ学習させる準備 ---
            # まず全パラメータを freeze（PEFTがある場合は既存のロジックのまま）
            # その後、埋め込みのうち新規トークン部分だけ勾配を許可する
            # （PEFTが None の場合 earlier loop set requires_grad=False, 再設定をここで行う）
            try:
                self.llm_embedder = self.llm.get_input_embeddings()
                # emb.weight は Parameter; 一旦全体を trainable にしておく（runner 側で古いトークンの勾配を消す）
                # まず既存ロジックで全パラメータ freeze している想定 → 明示的に埋め込みだけ True
                self.llm_embedder.weight.requires_grad = True
            except Exception as e:
                print("Warning: failed to set embedding requires_grad, exception:", e)
            # --- 追加: 追加トークンのみ学習させる準備 ---
            # まず全パラメータを freeze（PEFTがある場合は既存のロジックのまま）
            # その後、埋め込みのうち新規トークン部分だけ勾配を許可する
            # （PEFTが None の場合 earlier loop set requires_grad=False, 再設定をここで行う）
            #try:
            #    self.llm_embedder = self.llm.get_input_embeddings()
            #    # emb.weight は Parameter; 一旦全体を trainable にしておく（runner 側で古いトークンの勾配を消す）
            #    # まず既存ロジックで全パラメータ freeze している想定 → 明示的に埋め込みだけ True
            #    self.llm_embedder.weight.requires_grad = True
            #except Exception as e:
            #    print("Warning: failed to set embedding requires_grad, exception:", e)
            '''
            
            num_new_tokens = new_vocab_size - self.old_vocab_size
            print(f"Added {num_new_tokens} new tokens.")
            # 4. 埋め込みパラメータを取得
            self.embedding_layer = self.llm.get_input_embeddings()  # nn.Embedding

            for param in self.embedding_layer.parameters():
                param.requires_grad = False  # 一旦全部 True

            # 追加トークンの部分だけ optimizer に渡す場合は slice を指定
            self.new_token_params = [self.embedding_layer.weight[self.old_vocab_size:]]
            embedding_layer = self.llm.get_input_embeddings()
            self.new_weights = nn.Parameter(embedding_layer.weight[self.old_vocab_size:].clone())
            '''
            '''
            self.embedding_layer = self.llm.base_model.model.embed_tokens
            self.old_vocab_size = 32000

            # 追加トークン部分を独立した nn.Parameter にする
            self.new_token_weights = nn.Parameter(self.embedding_layer.weight.data[self.old_vocab_size:])
            nn.Parameter(self.embedding_layer.weight.data[self.old_vocab_size:]).requires_grad = True  # 念のため明示
            '''
            # ▼▼▼ 修正: 全ての準備が整った「この場所」で呼ぶ！ ▼▼▼
            self._init_concept_anchors()
            
            

            
            

            
            

            
            
            
            
            
            
            
        if load_flag:
            self.device = torch.device(device)
            print(f"Loading model from {model_weight_path}...")

    # ----------------------------------------------------------------
    # 1. トークナイザーの読み込み & トークン追加
    # ----------------------------------------------------------------
            self.tokenizer = AutoTokenizer.from_pretrained(model_weight_path + '/tokenizer')
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    # 論文で定義した6つの区切りトークン（必ずリストで定義）
            special_tokens_list = [
                "<|start_of_event|>", 
                "<|end_of_event|>", 
                "<|time_prefix|>", 
                "<|type_prefix|>", 
                "<|magnitude_prefix|>", 
                "<|depth_prefix|>"
            ]

    # トークナイザーに追加（保存済みなら追加されませんが、念のため）
            num_added = self.tokenizer.add_tokens(special_tokens_list)
    
    # ----------------------------------------------------------------
    # 2. ベースモデルの読み込み (AutoModel)
    # ----------------------------------------------------------------
            self.config = PeftConfig.from_pretrained(model_weight_path + '/model')
    
            self.llm = AutoModel.from_pretrained(
                self.config.base_model_name_or_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float32,
                device_map=self.device,
            )

    # ----------------------------------------------------------------
    # 3. 埋め込み層の拡張と「重みの復元」 (★最重要)
    # ----------------------------------------------------------------
    # まずサイズを拡張（この時点では新トークン部分はランダム初期化）
            self.llm.resize_token_embeddings(len(self.tokenizer))
    
    # ★ 保存しておいた追加トークンの重みファイルをロード
            token_weights_path = os.path.join(model_weight_path, 'new_token_weights.pt')
    
            if os.path.exists(token_weights_path):
                print(f"Loading trained token weights from {token_weights_path}...")
                new_token_weights = torch.load(token_weights_path, map_location=self.device)
        
        # モデルの埋め込み層を取得
                input_embeddings = self.llm.get_input_embeddings()
        
        # 埋め込み層の末尾（追加したトークン部分）に、学習済み重みをコピー
        # new_token_weights のサイズ分だけ後ろから上書きします
                num_new_tokens = new_token_weights.shape[0]
        
        # 形状チェックと代入
                with torch.no_grad():
                    input_embeddings.weight.data[-num_new_tokens:] = new_token_weights
            
                print("Token embeddings updated successfully.")
            else:
                print("Warning: new_token_weights.pt not found! New tokens are randomly initialized.")

    # ----------------------------------------------------------------
    # 4. LoRAアダプタの読み込み
    # ----------------------------------------------------------------
            self.llm = PeftModel.from_pretrained(
                self.llm,
                model_weight_path + '/model',
                torch_dtype=torch.float32,
                device_map=self.device,
            )

    # ----------------------------------------------------------------
    # 5. マージ (推論速度向上のため推奨)
    # ----------------------------------------------------------------
    # 検証・推論時はマージしてしまった方が計算が速く、扱いやすいです
    # ※ 再学習しないならここでマージしてOK
            #print("Merging LoRA adapter for inference...")
            #self.llm = self.llm.merge_and_unload()

            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False
        
            print("Model loaded and merged successfully.")
            self.llm_embedder = self.llm.get_input_embeddings()
            # ▼▼▼ 修正: 全ての準備が整った「この場所」で呼ぶ！ ▼▼▼
            self._init_concept_anchors()



        # Set the model parameters
        self.hidden_size = self.llm.config.hidden_size
        #self.llm_embedder = self.llm.get_input_embeddings()
        self.embedding_dim = self.llm_embedder.embedding_dim
        self.num_integral_samples = num_integral_samples
        self.num_event_types = num_event_types
        self.temporal_emb_type = temporal_emb_type
        self.temporal_emb_first = temporal_emb_first
        self.dtype = self.llm.dtype
        self.prompt = prompt

        """
        # old_vocab_size: 追加前の語彙数
        # embedding_dim: 埋め込みの次元数
        
        self.new_token_weights = nn.Parameter(
            torch.randn(
                len(self.tokenizer) - self.old_vocab_size,
                self.embedding_dim,
                device=self.llm_embedder.weight.device  # GPU 側に作成
            )
        )
        # 追加トークン部分だけ勾配を通す
        self.llm_embedder.weight.requires_grad_(True)
        print(self.llm_embedder.weight.requires_grad)  # True になる
        print(len(self.tokenizer))
        print(self.new_token_weights.requires_grad)  # True になる
        print(self.llm_embedder.weight.device)

        print(self.new_token_weights.device)
        """
        

        # 追加トークン部分だけ勾配を通す
        #self.llm_embedder.weight.requires_grad_(True)
        #print(self.llm_embedder.weight.requires_grad)  # True になる

        #self.magnitude_minmax_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)
        #self.depth_minmax_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)

        # Creat layers for computing intensities
        self.intensity_current = nn.Linear(
            1, self.num_event_types, bias=True, dtype=self.dtype, device=self.device)
        self.intensity_history = nn.Linear(
            self.hidden_size, self.num_event_types, bias=False, dtype=self.dtype, device=self.device)
        self.softplus = nn.Softplus()

        # Creat layers for predicting next events
        self.head_type = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_event_types, bias=True, dtype=self.dtype, device=self.device),
            nn.Softmax(dim=-1))
        self.head_time = nn.Linear(self.hidden_size, 1, bias=True, dtype=self.dtype, device=self.device)

        
        # --- (↓ ここからが新しいコード) ---
        if self.temporal_emb_type == 'positional':
            # 1. 時刻用のエンコーダー（学習可能なTPE）
            #dropout_rate = 0.1
            # 1. 個別のエンコーダ (Rawデータのみ使用: 入力次元=1)
            self.time_mlp = nn.Sequential(
                nn.Linear(1, 32), 
                nn.ReLU(), 
                #nn.Dropout(dropout_rate),
                nn.Linear(32, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)

            self.mag_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(), 
                #nn.Dropout(dropout_rate),
                nn.Linear(32, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)

            self.dep_mlp = nn.Sequential(
                nn.Linear(1, 32), 
                nn.ReLU(), 
                #nn.Dropout(dropout_rate),
                nn.Linear(32, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)

            
        elif self.temporal_emb_type == 'linear':
            # (もしlinearを使う場合も、3つに分離)
            self.temporal_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)
            self.magnitude_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)
            self.depth_embedder = nn.Linear(1, self.embedding_dim, dtype=self.dtype, device=self.device)
        
        elif self.temporal_emb_type == 'shifted':
            # (shiftedも同様に分離が必要だが、ここでは省略)
            # (shiftedは時刻と時間差の2入力が必要なので注意)
            raise NotImplementedError(f'Shifted type is not updated in this example.') 
        
        else:
            raise KeyError(f'Temporal embedding type {self.temporal_emb_type} not implemented.')
        '''
        self.embedder = LatLonMLPEncoding(embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.grad is not None)
        '''
        '''
        # ▼▼▼【重要】事前学習済み重みのロード処理 ▼▼▼
        if pretrain_path is not None and os.path.exists(pretrain_path):
            print(f"★ Loading pretrained MLP weights from: {pretrain_path}")
            try:
                # CPUでロードしてからデバイス転送するのが安全
                checkpoint = torch.load(pretrain_path, map_location='cpu')
                    
                self.time_mlp.load_state_dict(checkpoint['time_mlp'])
                self.mag_mlp.load_state_dict(checkpoint['mag_mlp'])
                self.dep_mlp.load_state_dict(checkpoint['dep_mlp'])
                self.gated_integration.load_state_dict(checkpoint['gated_integration'])
                    
                print(">> Successfully loaded pretrained weights for: time_mlp, mag_mlp, dep_mlp, gated_integration")
            except Exception as e:
                print(f"!! Error loading pretrained weights: {e}")
                print("Continuing with random initialization...")
        elif pretrain_path is not None:
            print(f"!! Warning: Pretrain path provided but file not found: {pretrain_path}")
        '''
                # ▲▲▲ ----------------------------------------- ▲▲▲
        # Generate the prompt embedding

        #with torch.no_grad():
        #    self.prompt_embeddings = self.embed_event_text(event_texts=[self.prompt], add_special_tokens=True)[0]
        
        #self.prompt_embeddings = self.embed_event_text(event_texts=[self.prompt], add_special_tokens=True)[0]
        # dropout 層をクラス内で定義（例: __init__ に追加）
        #self.dropout = torch.nn.Dropout(p=0.3)
        #self.dropout2 = torch.nn.Dropout(p=0.05)

        
        '''
        self.llm.save_pretrained(model_weight_path+'/model')
        self.tokenizer.save_pretrained(model_weight_path+'/tokenizer')
        print("保存しました")
        '''

    def _init_concept_anchors(self):
        """magnitude, depth, time の単語ベクトルを固定する"""
        print("Initializing Concept Anchors...")
        
        # ▼▼▼ 修正: 可視化や概念として代表的な表記（大文字始まり）に統一する ▼▼▼
        txt_mag = "Magnitude"  # magnitude -> Magnitude
        txt_dep = "Depth"      # depth -> Depth
        txt_time = "Time"      # time -> Time
        
        with torch.no_grad():
            # 平均を取って1つのベクトルにする
            vec_mag = self.embed_event_text([txt_mag])[0].mean(dim=0)
            vec_dep = self.embed_event_text([txt_dep])[0].mean(dim=0)
            vec_time = self.embed_event_text([txt_time])[0].mean(dim=0)

        # 固定バッファとして登録（学習しないのでこれでOK）
        self.register_buffer('anchor_mag', vec_mag)
        self.register_buffer('anchor_dep', vec_dep)
        self.register_buffer('anchor_time', vec_time)
        print("Anchors Initialized (Fixed to Word Embeddings).")

    
    
    
    
    
    
    
    def embed_event_text(self, event_texts: List[str], add_special_tokens: bool = False) -> List[Tensor]:
        """
        Embed the texts of events

        :param event_texts: event texts
        :param add_special_tokens: add special tokens or not
        :return: token embeddings of event texts, [(text_token_len, embedding_dim), ...]
        """
        event_tokens = self.tokenizer(
            event_texts, return_tensors='pt', add_special_tokens=add_special_tokens,
            padding=True, truncation=False)
        
        # トークンIDを表示
        #print("input_ids:\."+ event_tokens['input_ids'])
        """
        # それをトークンにデコードして確認する（オプション）
        f = open('tokenizer_watch.txt', 'a')
        for i, input_id in enumerate(event_tokens['input_ids']):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            
            f.write("Event"+str(i)+ "tokens:"+ str(tokens)+'\n')
            
        f.close()
        """
        
        nums_tokens = event_tokens['attention_mask'].to(self.device).sum(dim=-1)
        event_embeddings_padded = self.llm_embedder(event_tokens['input_ids'].to(self.device))
        event_embeddings = [
            event_embedding_padded[:num_tokens]
            for num_tokens, event_embedding_padded in zip(nums_tokens, event_embeddings_padded)]
        
        return event_embeddings
    
    '''
    def embed_event_text(self, event_texts: List[str], add_special_tokens: bool = False) -> List[Tensor]:
        """
        Embed event texts with proper gradient flow for new tokens (only new_token_weights are trainable)
    
        :param event_texts: List of event text strings
            :param add_special_tokens: Whether to add special tokens
        :return: List of token embeddings per event, each of shape (seq_len, embedding_dim)
        """
        # トークナイズ
        event_tokens = self.tokenizer(
            event_texts,
            return_tensors='pt',
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=False
        )

        input_ids = event_tokens['input_ids'].to(self.device)
        attention_mask = event_tokens['attention_mask'].to(self.device)
    
        # どのトークンが追加トークンかを判定
        added_mask = input_ids >= self.old_vocab_size  # shape: (batch, seq_len)
    
        
        # base_embeddings を勾配可能に
        base_embeddings = self.llm_embedder(input_ids)
        base_embeddings.requires_grad_(True)  # ← これがないと hook は登録できない

        # 勾配を 0.1 にスケール
        base_embeddings.register_hook(lambda grad: grad * 0.1)
    
        # 追加トークン用の埋め込みを scatter で安全に埋め込む
        new_embeddings = torch.zeros_like(base_embeddings, device=self.device)
        if added_mask.any():
            new_embeddings[added_mask] = self.new_token_weights[input_ids[added_mask] - self.old_vocab_size]
    
        # 元の埋め込みと追加トークン埋め込みを合成
        event_embeddings_padded = torch.where(
            added_mask.unsqueeze(-1),  # shape: (batch, seq_len, 1)
            new_embeddings,
            base_embeddings
        )   
    
        # attention_mask に応じて有効トークン長にスライス
        nums_tokens = attention_mask.sum(dim=-1)
        event_embeddings = [
            emb_padded[:num_tokens]
            for emb_padded, num_tokens in zip(event_embeddings_padded, nums_tokens)
        ]
    
        return event_embeddings
    '''

    
    
    def forward(
        self, batch_event_times: List[Tensor], batch_event_time_deltas: List[Tensor],
        batch_event_texts: List[List[str]],batch_event_mag:List[Tensor],batch_event_dep:List[Tensor]) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Forward function to get hidden states of event sequences in a batch

        :param batch_event_times: batch of event times in sequences, [(seq_len,), ...]
        :param batch_event_time_deltas: batch of event time deltas in sequences, [(seq_len,), ...]
        :param batch_event_texts: batch of event texts in sequences, [(seq_len,), ...]
        :return: hidden states of events, [(seq_len, hidden_size), ...]
        """
        batch_sequence_embeddings = []
        batch_attention_masks = []
        batch_event_emb_indices = []
        # ★ Loss計算用に全データを保存するリスト
        all_time_embs, all_mag_embs, all_dep_embs = [], [], []
        all_time_vals, all_mag_vals, all_dep_vals = [], [], []
        
        
        
        # Process each event sequence in the batch
        for event_times, event_time_deltas, event_texts,event_mag,event_dep  in zip(batch_event_times, batch_event_time_deltas,
                                                               batch_event_texts,batch_event_mag,batch_event_dep):
            
            # ▼▼▼ ここでプロンプト埋め込みを計算 ▼▼▼
            # ループ内で毎回計算グラフを作成する
            prompt_token_embeddings_tensor = self.embed_event_text(event_texts=[self.prompt], add_special_tokens=True)[0]
            # ▲▲▲ ---------------------------------- ▲▲▲
            
            sequence_embeddings = []
            sequence_attention_mask = []
            event_emb_indices = []

            im_start_texts1=["<|im_start|> system"]
            im_end_texts=["<|im_end|>"]
            im_start_texts2=["<|im_start|> sequence"]

            im_start_text_embeddings1= self.embed_event_text(im_start_texts1)
            im_end_text_embedding= self.embed_event_text(im_end_texts)
            im_start_text_embeddings2= self.embed_event_text(im_start_texts2)

            for im_start_text_embedding1 in im_start_text_embeddings1[0]:
                sequence_embeddings.append(im_start_text_embedding1)
                sequence_attention_mask.append(1)
            
            # Add the prompt embeddings
            for prompt_token_embedding in prompt_token_embeddings_tensor :
                sequence_embeddings.append(prompt_token_embedding)
                sequence_attention_mask.append(1)
                
            sequence_embeddings.append(im_end_text_embedding[0].squeeze(0))
            sequence_attention_mask.append(1)

            for im_start_text_embedding2 in im_start_text_embeddings2[0]:
                sequence_embeddings.append(im_start_text_embedding2)
                sequence_attention_mask.append(1)

            # Get the temporal embeddings for event times
            if self.temporal_emb_type == 'shifted':
                temporal_embeddings = self.temporal_embedder(
                    event_times.unsqueeze(-1), event_time_deltas.unsqueeze(-1))  # (seq_len, embedding_dim)
            else:
                # 緯度経度を0-1正規化
                '''
                event_texts1_norm = ((event_texts1 + 90.0) / 180.0).to(dtype=self.dtype, device=self.device)
                event_texts2_norm = ((event_texts2 + 180.0) / 360.0).to(dtype=self.dtype, device=self.device)
                '''
                #print(event_texts1[0])
                
                # 1. 個別のベクトル生成 (計算はする！統合に使うため)
                # ★ MLP実行
                temporal_embeddings = self.time_mlp(event_times.unsqueeze(-1))
                event_mag_embeddings = self.mag_mlp(event_mag.unsqueeze(-1))
                event_dep_embeddings = self.dep_mlp(event_dep.unsqueeze(-1))
            
                # ★ リスト保存
                all_time_embs.append(temporal_embeddings)
                all_mag_embs.append(event_mag_embeddings)
                all_dep_embs.append(event_dep_embeddings)
            
                # ★ 入力値も保存 (次元を合わせておく)
                all_time_vals.append(event_times.unsqueeze(-1))
                all_mag_vals.append(event_mag.unsqueeze(-1))
                all_dep_vals.append(event_dep.unsqueeze(-1))
                # 2. ★統合ベクトルの生成 (Conditioning)
                # Time, Mag, Dep をまとめて、TimeDeltaでGatingした「最強のベクトル」を作る
                #integrated_quake_vectors = self.gated_integration(
                #    temporal_embeddings, 
                #    event_mag_embeddings, 
                #    event_dep_embeddings, 
                #    event_time_deltas
                #) # (seq_len, dim)
                '''
                event_text_embeddings1=self.embedder(event_texts1_norm.unsqueeze(-1))
                event_text_embeddings2=self.embedder(event_texts2_norm.unsqueeze(-1))
                '''
            '''
            # Get token embeddings for the event texts
            event_text_embeddings = self.embed_event_text(event_texts)  # [(text_token_len, embedding_dim), ...]
            '''
            # Get token embeddings for the event texts
            #print(str(event_texts1))
            #print(str(type_event_str))
            
            start_event_texts=["<|start_of_event|>"]
            end_event_texts=["<|end_of_event|>"]
            time_event_texts=["<|time_prefix|>"]
            type_event_texts=["<|type_prefix|>"]
            mag_event_texts=["<|magnitude_prefix|>"]
            dep_event_texts=["<|depth_prefix|>"]
            # 統合ベクトル用のプレフィックス（Earthquake Prefix）
            # ※ もし <|earthquake_prefix|> がない場合は <|event_context|> などを使ってください
            #earth_event_text_embedding = self.embed_event_text(["<|earthquake_prefix|>"])
            
            
            start_event_text_embedding = self.embed_event_text(start_event_texts)
            end_event_text_embedding = self.embed_event_text(end_event_texts)
            time_event_text_embedding= self.embed_event_text(time_event_texts)
            type_event_text_embedding= self.embed_event_text(type_event_texts)
            mag_event_text_embedding= self.embed_event_text(mag_event_texts)
            dep_event_text_embedding= self.embed_event_text(dep_event_texts)
            
            
            
            #event_mag_embeddings = self.magnitude_minmax_embedder(event_mag.unsqueeze(-1))
            #event_dep_embeddings= self.depth_minmax_embedder(event_dep.unsqueeze(-1))
            event_text_embeddings = self.embed_event_text(event_texts)  # [(text_token_len, embedding_dim), ...]
            #event_text_embeddings1 = self.embed_event_text(event_texts1)
            #event_text_embeddings2=self.embed_event_text(event_texts2)
            #event_mag_embeddings=self.embed_event_text(event_mag)
            #event_dep_embeddings = self.embed_event_text(event_dep)  # [(text_token_len, embedding_dim), ...]
            # Process each event with event times and texts
            '''
            f = open('result/llama/0903_naiyou/a.txt', 'a')
            f.write(f"{event_mag[0]}\n")
            f.write(f"{event_mag_embeddings[0][0]}\n")
            f.write(f"{event_mag_embeddings[0][1]}\n")
            f.write(f"{event_mag_embeddings[0][2]}\n")
            f.write(f"{event_mag_embeddings[0][3]}\n")
            f.close()
            '''
            
            
            for temporal_embedding, event_token_embedding,event_mag_embedding,event_dep_embedding in zip(temporal_embeddings,event_text_embeddings,event_mag_embeddings,event_dep_embeddings):
                sequence_embeddings.append(start_event_text_embedding[0].squeeze(0))
                sequence_attention_mask.append(1)
                
                if self.temporal_emb_first:
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    sequence_embeddings.append(time_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)

                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    sequence_embeddings.append(type_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)
                    
                    sequence_embeddings.append(mag_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    
                    sequence_embeddings.append(event_mag_embedding)
                    sequence_attention_mask.append(1)

                    sequence_embeddings.append(dep_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    
                    sequence_embeddings.append(event_dep_embedding)
                    sequence_attention_mask.append(1)
                    
                    '''
                    for event_mag_embedding1 in event_mag_embedding:
                        sequence_embeddings.append(event_mag_embedding1)
                        sequence_attention_mask.append(1)
                    
                    
                    '''
                    #sequence_embeddings.append(event_mag_embedding)
                    #sequence_attention_mask.append(1)
                    #sequence_embeddings.append(dep_event_text_embedding[0].squeeze(0))
                    #sequence_attention_mask.append(1)
                    
                    #sequence_embeddings.append(event_dep_embedding)
                    #sequence_attention_mask.append(1)
                    '''
                    
                    for event_token_embedding11 in event_token_embedding1:
                        sequence_embeddings.append(event_token_embedding11)
                        sequence_attention_mask.append(1)
                    for event_token_embedding21 in event_token_embedding2:
                        sequence_embeddings.append(event_token_embedding21)
                        sequence_attention_mask.append(1)
                    '''

                    
                    
                    

                else:
                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    
                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    sequence_embeddings.append(type_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)
                    
                    sequence_embeddings.append(mag_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    
                    sequence_embeddings.append(event_mag_embedding)
                    sequence_attention_mask.append(1)

                    sequence_embeddings.append(dep_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    
                    sequence_embeddings.append(event_dep_embedding)
                    sequence_attention_mask.append(1)
                    #sequence_embeddings.append(event_mag_embedding)
                    #sequence_attention_mask.append(1)
                    #sequence_embeddings.append(event_dep_embedding)
                    #sequence_attention_mask.append(1)
                    
                    '''
                    for event_mag_embedding1 in event_mag_embedding:
                        sequence_embeddings.append(event_mag_embedding1)
                        sequence_attention_mask.append(1)
                    
                    for event_dep_embedding1 in event_dep_embedding:
                        sequence_embeddings.append(event_dep_embedding1)
                        sequence_attention_mask.append(1)
                    '''
                    
                    
                    
                    

                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    sequence_embeddings.append(time_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)
                # Record the index of the last embedding of this event
                sequence_embeddings.append(end_event_text_embedding[0].squeeze(0))
                sequence_attention_mask.append(1)
                event_emb_indices.append(len(sequence_embeddings) - 1)

            # Convert sequence embeddings to tensors
            batch_sequence_embeddings.append(torch.stack(sequence_embeddings))  # [(seq_emb_len, embedding_dim), ...]
            batch_attention_masks.append(torch.tensor(sequence_attention_mask).to(self.device))  # [(seq_emb_len,), ...]
            batch_event_emb_indices.append(torch.tensor(event_emb_indices).to(self.device))  # [(seq_len,), ...]

        # Pad the sequence embeddings in the batch
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_sequence_embeddings, batch_first=True)  # (num_seqs, max_seq_emb_len, embedding_dim)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks, batch_first=True)  # (num_seqs, max_seq_emb_len)
        
        # ★ここで Dropout を適用
        #padded_embeddings = self.dropout2(padded_embeddings)


        # Pass the padded embeddings through the LLM
        llm_output = self.llm(
            inputs_embeds=padded_embeddings, attention_mask=padded_attention_masks,
        ).last_hidden_state  # (batch_size, max_seq_emb_len, hidden_size)

        # Collect the hidden states at the last event embedding positions
        batch_hidden_states = [
            llm_output[i, batch_event_emb_indices[i], :]
            for i in range(len(batch_event_emb_indices))
        ]  # [(seq_len, hidden_size), ...]

        # -------------------------------------------------------
        # ★ 戻り値: 補助データ(aux_data)
        # -------------------------------------------------------
        aux_data = {
            'time_embs': torch.cat(all_time_embs, dim=0),
            'mag_embs':  torch.cat(all_mag_embs, dim=0),
            'dep_embs':  torch.cat(all_dep_embs, dim=0),
            'time_vals': torch.cat(all_time_vals, dim=0),
            'mag_vals':  torch.cat(all_mag_vals, dim=0),
            'dep_vals':  torch.cat(all_dep_vals, dim=0),
        }
        
        return batch_hidden_states, aux_data

    def compute_intensities(
        self, batch_event_time_deltas: List[Tensor], batch_hidden_states: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute intensities from event times and hidden states

        :param batch_event_time_deltas: event time deltas in a batch of event sequences, [(seq_len, n_samples), ...]
        :param batch_hidden_states: hidden states for a batch of event sequences, [(seq_len, hidden_size), ...]
        :return: a batch of event intensities, [(seq_len - 1, num_types), ...]
        """
        batch_intensities = []

        for event_time_deltas, hidden_states in zip(batch_event_time_deltas, batch_hidden_states):
            event_time_deltas_tensor = event_time_deltas.unsqueeze(1)[1:]  # (seq_len - 1, 1)
            intensities_current = self.intensity_current(event_time_deltas_tensor)  # (seq_len - 1, num_types)
            intensities_history = self.intensity_history(hidden_states[:-1])  # (seq_len - 1, num_types)
            #print("intensities_current.shape:", hidden_states[:-1].shape)
            #print("intensities_current.shape:", intensities_current.shape)
            #print("intensities_history.shape:", intensities_history.shape)
            intensities = self.softplus(intensities_current + intensities_history)  # (seq_len - 1, num_types)
            batch_intensities.append(intensities)  # [(seq_len - 1, num_types), ...]

        return batch_intensities

    def generate_time_deltas(self, batch_event_time_deltas: List[Tensor]) -> List[Tensor]:
        """
        Generate the time delta samples for every interval without padding.

        :param batch_event_time_deltas: list of event times since the last event, [(seq_len,), ...]
        :return: list of time samples for every interval, [(seq_len - 1, n_samples), ...]
        """
        batch_sampled_time_deltas = []

        # Process each sequence in the batch
        for event_time_deltas in batch_event_time_deltas:
            # Convert the tensor of time deltas
            event_time_deltas_tensor = event_time_deltas.unsqueeze(1)[1:]  # (seq_len - 1, 1)

            # Generate the sampling ratios
            time_delta_ratios = torch.linspace(
                start=0.0, end=1.0, steps=self.num_integral_samples, device=self.device)  # (1, n_samples)

            # Sample the time deltas across the intervals
            sampled_time_deltas = event_time_deltas_tensor * time_delta_ratios.unsqueeze(0)  # (seq_len - 1, n_samples)
            batch_sampled_time_deltas.append(sampled_time_deltas)  # [(seq_len - 1, n_samples), ...]

        return batch_sampled_time_deltas

    def compute_sampled_intensities(
        self, batch_sampled_time_deltas: List[Tensor], batch_hidden_states: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute intensities at sampled time deltas in a batch

        :param batch_sampled_time_deltas: a batch of sampled time delta sequence, [(seq_len - 1, n_samples), ...]
        :param batch_hidden_states: a batch of hidden state sequences, [(seq_len, hidden_size), ...]
        :return: a batch of intensities at sampled times, [(seq_len - 1, n_samples, num_types), ...]
        """
        batch_sampled_intensities = []

        for sampled_time_deltas, hidden_states in zip(batch_sampled_time_deltas, batch_hidden_states):
            sampled_time_deltas_tensor = sampled_time_deltas.unsqueeze(-1)  # (seq_len - 1, n_samples, 1)
            sampled_intensities_current = self.intensity_current(
                sampled_time_deltas_tensor)  # (seq_len - 1, n_samples, num_types)
            sampled_intensities_history = self.intensity_history(
                hidden_states[:-1]).unsqueeze(1)  # (seq_len - 1, 1, num_types)
            sampled_intensities = self.softplus(
                sampled_intensities_current + sampled_intensities_history)  # (seq_len - 1, n_samples, num_types)
            batch_sampled_intensities.append(sampled_intensities)  # [(seq_len - 1, n_samples, num_types), ...]

        return batch_sampled_intensities

    def compute_log_likelihood(
        self, batch_event_time_deltas: List[Tensor], batch_event_types: List[Tensor],
        batch_hidden_states: List[Tensor]) -> List[Tensor]:
        """
        Compute log likelihoods of event sequences in a batch

        :param batch_event_time_deltas: a batch of event time deltas, [(seq_len,), ...]
        :param batch_event_types: a batch of event types, [(seq_len,), ...]
        :param batch_hidden_states: a batch of hidden states, [(seq_len, hidden_size), ...]
        :return: a batch of log likelihoods, [(seq_len - 1,), ...]
        """
        # Compute intensities of events
        batch_event_intensities = self.compute_intensities(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1, num_types), ...]

        # Sample time deltas and compute their intensities
        batch_sampled_time_deltas = self.generate_time_deltas(
            batch_event_time_deltas=batch_event_time_deltas)  # [(seq_len - 1, n_samples), ...]
        batch_sampled_intensities = self.compute_sampled_intensities(
            batch_sampled_time_deltas=batch_sampled_time_deltas,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1, n_samples, num_types), ...]

        # Compute log likelihoods for the event part
        batch_log_likelihoods = []

        # event_types: (seq_len,), event_time_deltas: (seq_len,)
        # event_intensities: (seq_len - 1, num_types), sampled_intensities: (seq_len - 1, n_samples, num_types)
        for event_types, event_time_deltas, event_intensities, sampled_intensities in zip(
            batch_event_types, batch_event_time_deltas, batch_event_intensities, batch_sampled_intensities):
            # Compute the log likelihood part of events
            event_type_masks = functional.one_hot(
                event_types[1:], num_classes=self.num_event_types).to(self.device)  # (seq_len - 1, num_types)
            event_likelihoods = torch.sum(event_intensities * event_type_masks, dim=-1)  # (seq_len - 1,)
            event_log_likelihoods = torch.log(event_likelihoods)  # (seq_len - 1,)

            # Compute the log likelihood part of non-events
            sampled_total_intensities = torch.sum(sampled_intensities, dim=-1)  # (seq_len - 1, n_samples)
            non_event_log_likelihoods = \
                sampled_total_intensities.mean(dim=-1) * event_time_deltas[1:]  # (seq_len - 1,)

            log_likelihoods = event_log_likelihoods - non_event_log_likelihoods  # (seq_len - 1,)
            batch_log_likelihoods.append(log_likelihoods)

        return batch_log_likelihoods

    def compute_loss(self, batch: Dict[str, list]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the loss terms

        :param batch: a batch of event sequences
        :return: numbers of events, negative log likelihood (NLL) losses, event type prediction losses,
            event time prediction losses
        """
        
        batch_event_times = batch['time_since_start']
        batch_event_time_deltas = batch['time_since_last_event']
        batch_event_types = batch['type_event']
        batch_event_texts=batch['type_text']
        batch_event_mag = batch['magnitude']
        batch_event_dep = batch['depth']
        
        
        
    

        # Compute the hidden states
        batch_hidden_states, aux_data = self.forward(
            batch_event_times=batch_event_times,
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_texts=batch_event_texts,
            batch_event_mag=batch_event_mag,
            batch_event_dep=batch_event_dep)  # [(seq_len, hidden_size), ...]

        # Compute the log likelihoods
        batch_log_likelihoods = self.compute_log_likelihood(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_types=batch_event_types,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1,), ...]

        # Predict the next events
        batch_next_event_type_probs, batch_next_event_times = self.predict_next_event_probs(
            batch_hidden_states=batch_hidden_states)

        batch_nll_losses = []
        batch_type_losses = []
        batch_time_losses = []

        for log_likelihoods, next_event_type_probs, event_types, next_event_times, event_times in zip(
            batch_log_likelihoods, batch_next_event_type_probs, batch_event_types, batch_next_event_times,
            batch_event_times):
            nll_loss = - torch.sum(log_likelihoods, dim=0)
            event_type_masks = functional.one_hot(
                event_types[1:], num_classes=self.num_event_types).to(self.device)  # (seq_len - 1, num_types)
            
            
            type_loss = torch.sum(- event_type_masks * torch.log(next_event_type_probs[:-1]))
            time_loss = torch.sum((next_event_times[:-1] - event_times[1:]) ** 2, dim=0)

            batch_nll_losses.append(nll_loss)
            batch_type_losses.append(type_loss)
            batch_time_losses.append(time_loss)

        batch_event_nums = torch.LongTensor([len(event_times) - 1 for event_times in batch_event_times])
        batch_nll_losses = torch.stack(batch_nll_losses)
        batch_type_losses = torch.stack(batch_type_losses)
        batch_time_losses = torch.stack(batch_time_losses)

        # =======================================================
        # ★★★ ここに追加してください (自動初期化ロジック) ★★★
        # =======================================================
        # 生データ (aux_data['..._vals']) を使って、最初の1回だけscaleを自動調整します
        if self.training and not self.scale_initialized:
            # マグニチュード
            self._init_scale_param(aux_data['mag_vals'], self.scale_mag)
            # 深さ (これが特に重要！ 0.01~0.05くらいに自動設定されるはず)
            self._init_scale_param(aux_data['dep_vals'], self.scale_dep)
            # 時間
            self._init_scale_param(aux_data['time_vals'], self.scale_time)
            
            # フラグを更新して、次回以降は実行しないようにする
            self.scale_initialized = True
            
            # (オプション) 確認用ログ
            print(f"[Auto-Init] Scale Params Initialized:")
            print(f"  Mag scale : {self.scale_mag.item():.5f}")
            print(f"  Dep scale : {self.scale_dep.item():.5f}")
            print(f"  Time scale: {self.scale_time.item():.5f}")

        # -------------------------------------------------------
        # ★★★ 1. 概念アンカーLoss (Concept Alignment) ★★★
        # -------------------------------------------------------
        if self.alpha_concept > 0:
            target_mag = self.anchor_mag.unsqueeze(0).expand(aux_data['mag_embs'].size(0), -1)
            target_dep = self.anchor_dep.unsqueeze(0).expand(aux_data['dep_embs'].size(0), -1)
            target_time = self.anchor_time.unsqueeze(0).expand(aux_data['time_embs'].size(0), -1)

            loss_c_mag = F.mse_loss(aux_data['mag_embs'], target_mag)
            loss_c_dep = F.mse_loss(aux_data['dep_embs'], target_dep)
            loss_c_time = F.mse_loss(aux_data['time_embs'], target_time)
            
            total_concept_loss=(loss_c_mag + loss_c_dep + loss_c_time ) * self.alpha_concept
            #total_concept_loss=(loss_c_mag*self.magk + loss_c_dep*self.depk + loss_c_time*self.timek ) 
        else:
            # 0 なら計算せずに 0.0 のテンソルを作る（微分可能にしておく）
            total_concept_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        #loss_strong_anchors = (loss_c_mag + loss_c_dep) * self.alpha_concept
        #alpha_time = self.alpha_concept * 0.01
        #loss_weak_anchor = loss_c_time * alpha_time
        
        #total_concept_loss = loss_strong_anchors + loss_weak_anchor


        
        
        return batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses,total_concept_loss

    def predict_next_event_probs(self, batch_hidden_states: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Predict next events with probabilities (for training)

        :param batch_hidden_states: hidden states for a batch of event sequences, [(seq_len, hidden_size), ...]
        :return: a batch of next event type probabilities (in [(seq_len, num_types), ...])
            and next event times (in [(seq_len,), ...])
        """
        batch_next_event_type_probs = []
        batch_next_event_times = []

        for hidden_states in batch_hidden_states:
            next_event_type_probs = self.head_type(hidden_states)  # (seq_len, num_types)
            next_event_times = self.head_time(hidden_states).squeeze()  # (seq_len,)
            batch_next_event_type_probs.append(next_event_type_probs)  # [(seq_len, num_types), ...]
            batch_next_event_times.append(next_event_times)  # [(seq_len,), ...]

        return batch_next_event_type_probs, batch_next_event_times

    @torch.no_grad()
    def predict_next_events(self, batch: Dict[str, list]) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Predict next events (for evaluating)

        :param batch: a batch of event sequences
        :return: numbers of events, sequence log likelihoods, next event types, next event times
        """
        batch_event_times = batch['time_since_start']
        batch_event_time_deltas = batch['time_since_last_event']
        batch_event_types = batch['type_event']
        batch_event_texts=batch['type_text']
        batch_event_mag = batch['magnitude']
        batch_event_dep = batch['depth']

        
        # Compute the hidden states
        batch_hidden_states, _ = self.forward(
            batch_event_times=batch_event_times,
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_texts=batch_event_texts,
            batch_event_mag=batch_event_mag,
            batch_event_dep=batch_event_dep)  # [(seq_len, hidden_size), ...]
        
        # Compute the log likelihoods
        batch_log_likelihoods = self.compute_log_likelihood(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_types=batch_event_types,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1,), ...]
        batch_log_likelihood_sums = torch.stack(
            [log_likelihoods.sum(dim=-1) for log_likelihoods in batch_log_likelihoods])
        batch_event_nums = torch.LongTensor([len(event_times) - 1 for event_times in batch_event_times])

        # Predict the next events
        batch_next_event_type_probs, batch_next_event_times = self.predict_next_event_probs(
            batch_hidden_states=batch_hidden_states)
        batch_next_event_types = [
            next_event_type_probs.argmax(dim=-1) for next_event_type_probs in batch_next_event_type_probs]

        return batch_event_nums, batch_log_likelihood_sums, batch_next_event_types, batch_next_event_times
    
    def save_models(self, model_weight_path: str):
        """
        学習継続に影響を与えずに、LoRAアダプタと追加トークンを保存する関数
        """
        print(f"Saving adapter and tokens to {model_weight_path}...")

        os.makedirs(model_weight_path, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LoRAアダプタの保存 (マージしない！)
    # ---------------------------------------------------------
    # self.llm は PeftModel のままにしておく必要があります。
    # save_pretrained は LoRA の差分(adapter_model.bin)だけを保存してくれます。
        self.llm.save_pretrained(model_weight_path + '/model')

    # ---------------------------------------------------------
    # 2. 追加トークンの重みを個別に保存
    # ---------------------------------------------------------
    # モデル本体に混ぜ込まず、学習中の new_token_weights をそのまま保存します。
        if hasattr(self, 'new_token_weights') and self.new_token_weights is not None:
            torch.save(self.new_token_weights, os.path.join(model_weight_path, 'new_token_weights.pt'))
            print("New token weights saved separately.")

    # ---------------------------------------------------------
    # 3. トークナイザーの保存
    # ---------------------------------------------------------
        self.tokenizer.save_pretrained(model_weight_path + '/tokenizer')

        print("保存完了（学習は継続可能です）")

    