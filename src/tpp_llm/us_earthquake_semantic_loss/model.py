"""
TPP-LLM Model
"""
import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from peft import get_peft_model, PeftConfig, PeftModel 
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AddedToken

# Local Imports
from src.tpp_llm.us_earthquake_semantic_loss.layers import (
    TimePositionalEncoding, 
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class TPPLLMModel(nn.Module):
    """
    TPP-LLM Model
    """

    def __init__(
        self, model_name: str, num_event_types: int, num_integral_samples: int, temporal_emb_type: str,
        temporal_emb_first: bool = False, prompt: str = '', bnb_config: BitsAndBytesConfig = None,
        peft_config: PeftConfig = None, device: Union[str, torch.device] = 'cpu',
        model_weight_path:str ='',save_flag:bool=False,load_flag:bool=False,train_flag:bool=False,# ★追加: 事前学習済み重みのパスを受け取る引数
        beta_semantic: float = 1.0,  **kwargs):
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
        self.beta_semantic = beta_semantic

        if save_flag or train_flag:
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
            else:
                self.llm.eval()
                for param in self.llm.parameters():
                    param.requires_grad = False
            
            self.old_vocab_size = self.llm.get_input_embeddings().num_embeddings
            # Added: Flag to disable tokens when using TPE (positional)
            self.use_tokens = (temporal_emb_type == 'MLP')
            # Add special event tokens
            if self.use_tokens:
                event_tokens = [
            
                    AddedToken("<|start_of_event|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|end_of_event|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|time_prefix|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|type_prefix|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|magnitude_prefix|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|depth_prefix|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|im_start|>", special=True, lstrip=False, rstrip=False),
                    AddedToken("<|im_end|>", special=True, lstrip=False, rstrip=False),
                ]
                self.old_vocab_size = len(self.tokenizer)
                self.tokenizer.add_tokens(event_tokens, special_tokens=True)
                self.llm.resize_token_embeddings(len(self.tokenizer))
            
                print(f"[TPPLLMModel] old_vocab_size={getattr(self,'old_vocab_size', None)}, "
                    f"total_vocab={len(self.tokenizer)}, "
                    f"embedding_dim={self.llm.get_input_embeddings().embedding_dim}")
            
            # Allow gradients only for the newly added token embeddings
                try:
                    self.llm_embedder = self.llm.get_input_embeddings()
                    self.llm_embedder.weight.requires_grad = True
                except Exception as e:
                    print("Warning: failed to set embedding requires_grad, exception:", e)
            self.llm_embedder = self.llm.get_input_embeddings()
            # Initialize concept anchors
            self._init_concept_anchors()
            
        if load_flag:
            self.device = torch.device(device)
            print(f"Loading model from {model_weight_path}...")

            # Load tokenizer and add special tokens
            self.tokenizer = AutoTokenizer.from_pretrained(model_weight_path + '/tokenizer')
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
            # Load base model configuration
            self.config = PeftConfig.from_pretrained(model_weight_path + '/model')
            self.llm = AutoModel.from_pretrained(
                self.config.base_model_name_or_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float32,
                device_map=self.device,
            )

            # Resize token embeddings and restore weights for new tokens
            self.llm.resize_token_embeddings(len(self.tokenizer))
            token_weights_path = os.path.join(model_weight_path, 'new_token_weights.pt')
    
            if os.path.exists(token_weights_path):
                print(f"Loading trained token weights from {token_weights_path}...")
                new_token_weights = torch.load(token_weights_path, map_location=self.device)
                input_embeddings = self.llm.get_input_embeddings()
                num_new_tokens = new_token_weights.shape[0]
        
                with torch.no_grad():
                    input_embeddings.weight.data[-num_new_tokens:] = new_token_weights
                print("Token embeddings updated successfully.")
            else:
                print("Warning: new_token_weights.pt not found! New tokens are randomly initialized.")

            # Load LoRA adapter
            self.llm = PeftModel.from_pretrained(
                self.llm,
                model_weight_path + '/model',
                torch_dtype=torch.float32,
                device_map=self.device,
            )

            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False
        
            print("Model loaded and merged successfully.")
            self.llm_embedder = self.llm.get_input_embeddings()

            # Initialize concept anchors
            self._init_concept_anchors()



        # Set the model parameters
        self.hidden_size = self.llm.config.hidden_size
        self.embedding_dim = self.llm_embedder.embedding_dim
        self.num_integral_samples = num_integral_samples
        self.num_event_types = num_event_types
        self.temporal_emb_type = temporal_emb_type
        self.temporal_emb_first = temporal_emb_first
        self.dtype = self.llm.dtype
        self.prompt = prompt

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

        
        # Value to Vector Encoders (MLPs)
        if self.temporal_emb_type == 'MLP':
            self.time_mlp = nn.Sequential(
                nn.Linear(1, 32), 
                nn.ReLU(), 
                nn.Linear(32, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)

            self.mag_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(), 
                nn.Linear(32, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)

            self.dep_mlp = nn.Sequential(
                nn.Linear(1, 32), 
                nn.ReLU(), 
                nn.Linear(32, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)

        elif self.temporal_emb_type =='positional':
            self.temporal_embedder = TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
        
        else:
            raise KeyError(f'Temporal embedding type {self.temporal_emb_type} not implemented.')

    def _init_concept_anchors(self):
        """
        Initialize and freeze the semantic anchors based on the word embeddings of specific concepts.
        """
        print("Initializing Concept Anchors...")
        
        txt_mag = "Magnitude"  
        txt_dep = "Depth"      
        txt_time = "Time"      
        
        with torch.no_grad():
            vec_mag = self.embed_event_text([txt_mag])[0].mean(dim=0)
            vec_dep = self.embed_event_text([txt_dep])[0].mean(dim=0)
            vec_time = self.embed_event_text([txt_time])[0].mean(dim=0)

        # Register as fixed buffers (no gradients will be computed)
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
        
        nums_tokens = event_tokens['attention_mask'].to(self.device).sum(dim=-1)
        event_embeddings_padded = self.llm_embedder(event_tokens['input_ids'].to(self.device))
        event_embeddings = [
            event_embedding_padded[:num_tokens]
            for num_tokens, event_embedding_padded in zip(nums_tokens, event_embeddings_padded)]
        
        return event_embeddings
    
    def forward(
        self, batch_event_times: List[Tensor], 
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

        # Lists to save data for loss computation
        all_time_embs, all_mag_embs, all_dep_embs = [], [], []
        all_time_vals, all_mag_vals, all_dep_vals = [], [], []
        aux_data=[]
        
        self.use_tokens=(self.temporal_emb_type== 'MLP')

        # Process each event sequence in the batch
        for event_times, event_texts,event_mag,event_dep  in zip(batch_event_times,
                                                               batch_event_texts,batch_event_mag,batch_event_dep):
            
            prompt_token_embeddings_tensor = self.embed_event_text(event_texts=[self.prompt], add_special_tokens=True)[0]
            
            sequence_embeddings = []
            sequence_attention_mask = []
            event_emb_indices = []

            if self.use_tokens:
                im_start_texts1=["<|im_start|> system"]
                im_end_texts=["<|im_end|>"]
                im_start_texts2=["<|im_start|> sequence"]
                start_event_texts=["<|start_of_event|>"]
                end_event_texts=["<|end_of_event|>"]
                time_event_texts=["<|time_prefix|>"]
                type_event_texts=["<|type_prefix|>"]
                mag_event_texts=["<|magnitude_prefix|>"]
                dep_event_texts=["<|depth_prefix|>"]
            
            if self.use_tokens:
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

            if self.use_tokens:    
                sequence_embeddings.append(im_end_text_embedding[0].squeeze(0))
                sequence_attention_mask.append(1)

                for im_start_text_embedding2 in im_start_text_embeddings2[0]:
                    sequence_embeddings.append(im_start_text_embedding2)
                    sequence_attention_mask.append(1)

            # Get the temporal embeddings for event times
            if self.temporal_emb_type == 'MLP':
                temporal_embeddings = self.time_mlp(event_times.unsqueeze(-1))
                event_mag_embeddings = self.mag_mlp(event_mag.unsqueeze(-1))
                event_dep_embeddings = self.dep_mlp(event_dep.unsqueeze(-1))
            
                all_time_embs.append(temporal_embeddings)
                all_mag_embs.append(event_mag_embeddings)
                all_dep_embs.append(event_dep_embeddings)
            
                all_time_vals.append(event_times.unsqueeze(-1))
                all_mag_vals.append(event_mag.unsqueeze(-1))
                all_dep_vals.append(event_dep.unsqueeze(-1))
                
            else:
                temporal_embeddings = self.temporal_embedder(event_times.unsqueeze(-1))  # (seq_len, embedding_dim)
                event_mag_embeddings = self.temporal_embedder(event_mag.unsqueeze(-1))  # (seq_len, embedding_dim)
                event_dep_embeddings = self.temporal_embedder(event_dep.unsqueeze(-1))  # (seq_len, embedding_dim)
            
            
            if self.use_tokens:
                start_event_text_embedding = self.embed_event_text(start_event_texts)
                end_event_text_embedding = self.embed_event_text(end_event_texts)
                time_event_text_embedding= self.embed_event_text(time_event_texts)
                type_event_text_embedding= self.embed_event_text(type_event_texts)
                mag_event_text_embedding= self.embed_event_text(mag_event_texts)
                dep_event_text_embedding= self.embed_event_text(dep_event_texts)
            
            event_text_embeddings = self.embed_event_text(event_texts)  # [(text_token_len, embedding_dim), ...]
            
            for temporal_embedding, event_token_embedding,event_mag_embedding,event_dep_embedding in zip(temporal_embeddings,event_text_embeddings,event_mag_embeddings,event_dep_embeddings):
                if self.use_tokens:
                    sequence_embeddings.append(start_event_text_embedding[0].squeeze(0))
                    sequence_attention_mask.append(1)
                
                if self.temporal_emb_first:
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(time_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)

                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)

                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(type_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)

                    if self.use_tokens:
                        sequence_embeddings.append(mag_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    sequence_embeddings.append(event_mag_embedding)
                    sequence_attention_mask.append(1)

                    if self.use_tokens:
                        sequence_embeddings.append(dep_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    sequence_embeddings.append(event_dep_embedding)
                    sequence_attention_mask.append(1)

                else:
                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(type_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)
                    
                    if self.use_tokens:
                        sequence_embeddings.append(mag_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    sequence_embeddings.append(event_mag_embedding)
                    sequence_attention_mask.append(1)

                    if self.use_tokens:
                        sequence_embeddings.append(dep_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    sequence_embeddings.append(event_dep_embedding)
                    sequence_attention_mask.append(1)
                
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(time_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)
                # Record the index of the last embedding of this event
                if self.use_tokens:
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
        
        # Pass the padded embeddings through the LLM
        llm_output = self.llm(
            inputs_embeds=padded_embeddings, attention_mask=padded_attention_masks,
        ).last_hidden_state  # (batch_size, max_seq_emb_len, hidden_size)

        # Collect the hidden states at the last event embedding positions
        batch_hidden_states = [
            llm_output[i, batch_event_emb_indices[i], :]
            for i in range(len(batch_event_emb_indices))
        ]  # [(seq_len, hidden_size), ...]


        if self.temporal_emb_type == 'MLP':
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
            event_type_masks = F.one_hot(
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
            event_type_masks = F.one_hot(
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

        # 1. Concept Alignment Loss
        if self.beta_semantic > 0 and self.temporal_emb_type == 'MLP':
            target_mag = self.anchor_mag.unsqueeze(0).expand(aux_data['mag_embs'].size(0), -1)
            target_dep = self.anchor_dep.unsqueeze(0).expand(aux_data['dep_embs'].size(0), -1)
            target_time = self.anchor_time.unsqueeze(0).expand(aux_data['time_embs'].size(0), -1)

            loss_c_mag = F.mse_loss(aux_data['mag_embs'], target_mag)
            loss_c_dep = F.mse_loss(aux_data['dep_embs'], target_dep)
            loss_c_time = F.mse_loss(aux_data['time_embs'], target_time)
        
            raw_semantic_loss = loss_c_mag + loss_c_dep + loss_c_time
        else:
            raw_semantic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses,raw_semantic_loss

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
        Save the LoRA adapter and additional token weights without affecting training continuation.
        """
        print(f"Saving adapter and tokens to {model_weight_path}...")

        os.makedirs(model_weight_path, exist_ok=True)
    
        # 1. Save LoRA adapter (without merging)
        self.llm.save_pretrained(model_weight_path + '/model')

        # 2. Save new token weights separately
        if hasattr(self, 'new_token_weights') and self.new_token_weights is not None:
            torch.save(self.new_token_weights, os.path.join(model_weight_path, 'new_token_weights.pt'))
            print("New token weights saved separately.")

        # 3. Save tokenizer
        self.tokenizer.save_pretrained(model_weight_path + '/tokenizer')

        print("Save complete (Training can be continued).")

    