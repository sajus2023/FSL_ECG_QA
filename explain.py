import argparse
import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.io as sio
from typing import Dict, List, Optional, Tuple
import math
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from load_class import prepare_ecg_qa_data
from data_loader import FSL_ECG_QA_DataLoader
from meta_trainer import MetaTrainer
from utils import set_device

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

device = set_device()

# ============================================================================
# GRADIENT-BASED SALIENCY METHODS
# ============================================================================

def compute_end_to_end_saliency(meta_learner, ecg_input, question_tokens, question_mask, answer_mask):
    """
    Compute saliency through the entire pipeline:
    ECG → Encoder → Mapper → LLM → Output

    Args:
        meta_learner: The MetaLearner model
        ecg_input: [batch, 12, 5000] ECG tensor
        question_tokens: Tokenized question
        question_mask: Question attention mask
        answer_mask: Answer attention mask

    Returns:
        saliency: [5000] importance scores for each ECG timepoint
    """
    meta_learner.eval()
    dev = next(meta_learner.parameters()).device

    # Ensure ECG requires gradients
    ecg_input = ecg_input.to(dev).requires_grad_(True)

    # Step 1: ECG Encoder forward
    padding_mask = torch.zeros(ecg_input.shape[0], ecg_input.shape[1], ecg_input.shape[2], dtype=torch.bool, device=dev)
    encoder_out = meta_learner.feature_extract(source=ecg_input, padding_mask=padding_mask)

    # Get encoder features
    if isinstance(encoder_out, dict):
        ecg_features = encoder_out.get('encoder_out', encoder_out.get('features'))
    else:
        ecg_features = encoder_out

    # Pool if needed
    if ecg_features.dim() == 3:
        ecg_features = ecg_features.mean(dim=1)

    # Step 2: Mapper forward
    fast_weights = list(meta_learner.mapper_net.parameters())
    prefix_embeddings = meta_learner.mapper_net(ecg_features, fast_weights)

    # Target: L2 norm of prefix embeddings
    target = prefix_embeddings.norm()

    # Backpropagate
    target.backward()

    # Get saliency from ECG input gradients
    saliency = ecg_input.grad.abs().squeeze(0)  # [12, 5000]

    # Average across leads
    saliency_avg = saliency.mean(dim=0)  # [5000]

    return saliency_avg.detach().cpu()

def compute_answer_driven_saliency(meta_learner, ecg_input, question_tokens, question_mask, answer_mask, fast_weights=None):
    """
    Compute saliency driven by actual answer generation.
    Shows which ECG regions most affect the predicted answer.
    """
    meta_learner.eval()
    dev = next(meta_learner.parameters()).device

    # Take first sample if batched
    if ecg_input.dim() == 3 and ecg_input.shape[0] > 1:
        ecg_input = ecg_input[0:1]
        question_tokens = question_tokens[0:1]
        question_mask = question_mask[0:1]
        answer_mask = answer_mask[0:1]

    ecg_input = ecg_input.to(dev).requires_grad_(True)
    question_tokens = question_tokens.to(dev)
    question_mask = question_mask.to(dev)
    answer_mask = answer_mask.to(dev)

    # Forward pass through entire model
    if fast_weights is None:
        fast_weights = list(meta_learner.mapper_net.parameters())

    # Get logits
    logits, pred_tokens = meta_learner(
        ecg_input,
        question_tokens,
        question_mask,
        answer_mask,
        fast_weights,
        get_pred_tokens=True
    )

    # Target: maximize confidence of predicted tokens
    target = logits.max(dim=-1).values.mean()

    # Backpropagate
    meta_learner.zero_grad()
    target.backward()

    # Get saliency
    if ecg_input.grad is not None:
        saliency = ecg_input.grad.abs().squeeze(0)  # [12, 5000]
        saliency_avg = saliency.mean(dim=0)  # [5000]
        return saliency_avg.detach().cpu()
    else:
        print("Warning: No gradients computed. Check if model is in eval mode with gradients enabled.")
        return None

def compute_component_saliency(meta_learner, ecg_input):
    """
    Compute saliency at different stages to compare:
    1. Encoder-only saliency
    2. Encoder + Mapper saliency
    """
    dev = next(meta_learner.parameters()).device
    results = {}

    # === 1. Encoder-only saliency ===
    ecg_input_enc = ecg_input.clone().to(dev).requires_grad_(True)
    padding_mask = torch.zeros(ecg_input_enc.shape[0], ecg_input_enc.shape[1], ecg_input_enc.shape[2], dtype=torch.bool, device=dev)

    encoder_out = meta_learner.feature_extract(source=ecg_input_enc, padding_mask=padding_mask)
    if isinstance(encoder_out, dict):
        features = encoder_out.get('encoder_out', encoder_out.get('features'))
    else:
        features = encoder_out

    target_enc = features.mean()
    target_enc.backward()

    saliency_enc = ecg_input_enc.grad.abs().squeeze(0).mean(dim=0)
    results['encoder_only'] = saliency_enc.detach().cpu()

    # === 2. Encoder + Mapper saliency ===
    ecg_input_map = ecg_input.clone().to(dev).requires_grad_(True)
    padding_mask = torch.zeros(ecg_input_map.shape[0], ecg_input_map.shape[1], ecg_input_map.shape[2], dtype=torch.bool, device=dev)

    encoder_out = meta_learner.feature_extract(source=ecg_input_map, padding_mask=padding_mask)
    if isinstance(encoder_out, dict):
        features = encoder_out.get('encoder_out', encoder_out.get('features'))
    else:
        features = encoder_out

    # Pool if needed
    if features.dim() == 3:
        features = features.mean(dim=1)

    # Through mapper
    fast_weights = list(meta_learner.mapper_net.parameters())
    prefix = meta_learner.mapper_net(features, fast_weights)
    target_map = prefix.mean()
    target_map.backward()

    saliency_map = ecg_input_map.grad.abs().squeeze(0).mean(dim=0)
    results['encoder_mapper'] = saliency_map.detach().cpu()

    return results

# ============================================================================
# ATTENTION EXTRACTION METHODS
# ============================================================================
class AttentionExtractor:
    """
    Extract attention weights from different components of the model.
    """

    def __init__(self, meta_learner):
        self.meta_learner = meta_learner
        self.device = next(meta_learner.parameters()).device
        self._hooks = []
        self._attention_cache = {}

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_cache = {}

    def extract_mapper_attention(self, ecg_input: torch.Tensor, fast_weights: Optional[List] = None) -> torch.Tensor:
        """
        Extract attention weights from the AttentionMapper.

        The mapper computes: A = softmax(Q @ K.T / sqrt(d))
        where Q, K come from [prefix; ecg_features]

        Args:
            ecg_input: [batch, 12, 5000] ECG tensor
            fast_weights: Optional adapted weights

        Returns:
            attention: [batch, num_heads, prefix_len+1, prefix_len+1] attention weights
        """
        self.meta_learner.eval()
        ecg_input = ecg_input.to(self.device)

        with torch.no_grad():
            # Get ECG features through encoder
            padding_mask = torch.zeros(ecg_input.shape[0], ecg_input.shape[1], ecg_input.shape[2],
                                       dtype=torch.bool, device=self.device)
            encoder_out = self.meta_learner.feature_extract(source=ecg_input, padding_mask=padding_mask)

            if isinstance(encoder_out, dict):
                clip_prefix = encoder_out.get('encoder_out', encoder_out.get('features'))
            else:
                clip_prefix = encoder_out

            # Pool if 3D
            if clip_prefix.dim() == 3:
                clip_prefix = clip_prefix.mean(dim=1)

            clip_prefix = clip_prefix.to(self.device)

            # Get mapper weights
            if fast_weights is None:
                fast_weights = list(self.meta_learner.mapper_net.parameters())

            # Replicate mapper forward to extract attention
            clip_x = clip_prefix.float().unsqueeze(1)
            batch_size = clip_x.shape[0]

            prefix = fast_weights[0].unsqueeze(0).expand(batch_size, *fast_weights[0].shape)
            x_prefix = torch.cat((prefix, clip_x), dim=1)  # [batch, prefix_len+1, dim]

            # Compute Q, K
            from torch.nn import functional as F
            Q = F.linear(x_prefix, weight=fast_weights[1], bias=fast_weights[2])
            K = F.linear(x_prefix, weight=fast_weights[3], bias=fast_weights[4])

            dim_V = self.meta_learner.mapper_net.dim_V
            num_heads = self.meta_learner.mapper_net.num_heads
            dim_split = dim_V // num_heads

            # Split into heads
            Q_ = torch.cat(Q.split(dim_split, 2), 0)  # [batch*num_heads, seq, dim_head]
            K_ = torch.cat(K.split(dim_split, 2), 0)

            # Compute attention weights
            A = F.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_V), dim=2)

            # Reshape to [batch, num_heads, seq, seq]
            A = torch.stack(A.split(batch_size, 0), dim=1)

        return A

    def extract_llm_attention(self, ecg_input: torch.Tensor, question_tokens: torch.Tensor,
                               question_mask: torch.Tensor, answer_mask: torch.Tensor,
                               fast_weights: Optional[List] = None,
                               layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """
        Extract attention weights from LLaMA layers.

        Args:
            ecg_input: [batch, 12, 5000] ECG tensor
            question_tokens: [batch, seq_len] tokenized question
            question_mask: [batch, seq_len] question mask
            answer_mask: [batch, seq_len] answer mask
            fast_weights: Optional adapted weights
            layers: List of layer indices to extract (default: all)

        Returns:
            Dict mapping layer_idx -> attention tensor [batch, num_heads, seq, seq]
        """
        self.meta_learner.eval()
        self._attention_cache = {}

        # Get total number of layers
        num_layers = self.meta_learner.gpt.config.num_hidden_layers
        if layers is None:
            layers = list(range(num_layers))

        # Register hooks on attention layers
        def make_hook(layer_idx):
            def hook(module, input, output):
                # LLaMA attention output is (attn_output, attn_weights, past_key_value)
                # But by default attn_weights is None unless output_attentions=True
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self._attention_cache[layer_idx] = output[1].detach().cpu()
            return hook

        # Register hooks
        for layer_idx in layers:
            layer = self.meta_learner.gpt.model.layers[layer_idx]
            hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(hook)

        ecg_input = ecg_input.to(self.device)
        question_tokens = question_tokens.to(self.device)
        question_mask = question_mask.to(self.device)
        answer_mask = answer_mask.to(self.device)

        if fast_weights is None:
            fast_weights = list(self.meta_learner.mapper_net.parameters())

        with torch.no_grad():
            # Build input embeddings
            padding_mask = torch.zeros(ecg_input.shape[0], ecg_input.shape[1], ecg_input.shape[2],
                                       dtype=torch.bool, device=self.device)
            proj_clip = self.meta_learner.feature_extract(source=ecg_input, padding_mask=padding_mask)
            clip_prefix = proj_clip['encoder_out'].to(self.device)

            if clip_prefix.dim() == 3:
                clip_prefix = clip_prefix.mean(dim=1)

            tokens_embed = self.meta_learner.get_gpt_embeddings(question_tokens).to(self.device)
            proj_clip = self.meta_learner.mapper_net(clip_prefix, fast_weights).to(self.device)
            embedding_cat = torch.cat((proj_clip, tokens_embed), dim=1)

            # Forward through LLaMA with output_attentions=True
            out = self.meta_learner.gpt(inputs_embeds=embedding_cat, output_attentions=True)

            # Extract attentions from output (more reliable than hooks for LLaMA)
            if hasattr(out, 'attentions') and out.attentions is not None:
                for layer_idx in layers:
                    if layer_idx < len(out.attentions):
                        self._attention_cache[layer_idx] = out.attentions[layer_idx].detach().cpu()

        # Clean up hooks
        self.clear_hooks()

        return self._attention_cache

    def extract_ecg_encoder_attention(self, ecg_input: torch.Tensor,
                                       layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """
        Extract attention weights from ECG-FM encoder transformer layers.

        Args:
            ecg_input: [batch, 12, 5000] ECG tensor
            layers: List of layer indices (default: all 12 layers)

        Returns:
            Dict mapping layer_idx -> attention tensor
        """
        self.meta_learner.eval()
        self._attention_cache = {}

        # ECG-FM has 12 transformer layers
        if layers is None:
            layers = list(range(12))

        ecg_input = ecg_input.to(self.device)

        # Try to access the internal encoder
        encoder = self.meta_learner.feature_extract
        if hasattr(encoder, 'model'):
            encoder = encoder.model

        # Register hooks on transformer layers
        def make_hook(layer_idx):
            def hook(module, input, output):
                # fairseq-signals attention returns (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    self._attention_cache[layer_idx] = output[1].detach().cpu()
            return hook

        # Try to find encoder layers
        try:
            if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layers'):
                encoder_layers = encoder.encoder.layers
            elif hasattr(encoder, 'layers'):
                encoder_layers = encoder.layers
            else:
                print("Warning: Could not find encoder layers for attention extraction")
                return {}

            for layer_idx in layers:
                if layer_idx < len(encoder_layers):
                    layer = encoder_layers[layer_idx]
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
                        self._hooks.append(hook)
        except Exception as e:
            print(f"Warning: Could not register hooks on ECG encoder: {e}")
            return {}

        with torch.no_grad():
            padding_mask = torch.zeros(ecg_input.shape[0], ecg_input.shape[1], ecg_input.shape[2],
                                       dtype=torch.bool, device=self.device)
            _ = self.meta_learner.feature_extract(source=ecg_input, padding_mask=padding_mask)

        self.clear_hooks()

        return self._attention_cache

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def visualize_component_comparison(ecg_signal, saliency_dict, lead_idx=1, fs=500,
                                    question=None, answer=None, save_path=None):
    """
    Visualize saliency comparison between encoder-only and encoder+mapper.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    ecg = ecg_signal[lead_idx].cpu().numpy()
    time = np.arange(len(ecg)) / fs

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Plot 1: Raw ECG
    axes[0].plot(time, ecg, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].set_title(f'ECG Lead {lead_names[lead_idx]}')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Encoder-only saliency
    sal_enc = saliency_dict['encoder_only'].numpy()
    sal_enc_smooth = gaussian_filter1d(sal_enc, sigma=50)
    sal_enc_norm = (sal_enc_smooth - sal_enc_smooth.min()) / (sal_enc_smooth.max() - sal_enc_smooth.min() + 1e-8)

    axes[1].fill_between(time, 0, sal_enc_norm, alpha=0.7, color='blue', label='Encoder Only')
    axes[1].set_ylabel('Importance')
    axes[1].set_title('ECG Encoder Saliency (what encoder focuses on)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Encoder + Mapper saliency
    sal_map = saliency_dict['encoder_mapper'].numpy()
    sal_map_smooth = gaussian_filter1d(sal_map, sigma=50)
    sal_map_norm = (sal_map_smooth - sal_map_smooth.min()) / (sal_map_smooth.max() - sal_map_smooth.min() + 1e-8)

    axes[2].fill_between(time, 0, sal_map_norm, alpha=0.7, color='red', label='Encoder + Mapper')
    axes[2].set_ylabel('Importance')
    axes[2].set_title('Encoder + Mapper Saliency (what mapper projects to LLM)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Plot 4: Comparison overlay
    axes[3].fill_between(time, 0, sal_enc_norm, alpha=0.5, color='blue', label='Encoder Only')
    axes[3].fill_between(time, 0, sal_map_norm, alpha=0.5, color='red', label='Encoder + Mapper')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Importance')
    axes[3].set_title('Saliency Comparison')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    if question and answer:
        fig.suptitle(f"Q: {question}\nA: {answer}", fontsize=11, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.close(fig)
    return fig

def visualize_mapper_attention(attention: torch.Tensor, prefix_length: int = 4,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize AttentionMapper attention weights.

    Shows how the learnable prefix tokens attend to the ECG feature.

    Args:
        attention: [batch, num_heads, seq, seq] attention weights
        prefix_length: Number of prefix tokens
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Take first sample, average over heads
    attn = attention[0].mean(dim=0).cpu().numpy()  # [seq, seq]

    seq_len = attn.shape[0]
    labels = [f'P{i}' for i in range(prefix_length)] + ['ECG']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full attention matrix
    im1 = axes[0].imshow(attn, cmap='Blues', aspect='auto')
    axes[0].set_xticks(range(seq_len))
    axes[0].set_yticks(range(seq_len))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel('Key (attended to)')
    axes[0].set_ylabel('Query (attending from)')
    axes[0].set_title('Mapper Attention Matrix\n(averaged over heads)')
    plt.colorbar(im1, ax=axes[0])

    # How much each prefix token attends to ECG feature
    ecg_attention = attn[:prefix_length, -1]  # Prefix tokens attending to ECG
    axes[1].bar(range(prefix_length), ecg_attention, color='steelblue')
    axes[1].set_xticks(range(prefix_length))
    axes[1].set_xticklabels([f'Prefix {i}' for i in range(prefix_length)])
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('How Prefix Tokens Attend to ECG Feature')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mapper attention to: {save_path}")

    plt.close(fig)
    return fig

def visualize_llm_attention_to_prefix(llm_attentions: Dict[int, torch.Tensor],
                                       prefix_length: int = 4,
                                       layers_to_plot: Optional[List[int]] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize how LLaMA attends to the ECG prefix tokens across layers.

    Args:
        llm_attentions: Dict mapping layer_idx -> [batch, heads, seq, seq]
        prefix_length: Number of ECG prefix tokens
        layers_to_plot: Which layers to visualize (default: first, middle, last)
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if not llm_attentions:
        print("No LLM attention weights available")
        return None

    available_layers = sorted(llm_attentions.keys())

    if layers_to_plot is None:
        # Select first, middle, and last layers
        n = len(available_layers)
        layers_to_plot = [available_layers[0], available_layers[n//2], available_layers[-1]]

    fig, axes = plt.subplots(2, len(layers_to_plot), figsize=(5*len(layers_to_plot), 10))

    # Handle case where only 1 layer
    if len(layers_to_plot) == 1:
        axes = axes.reshape(2, 1)

    for col, layer_idx in enumerate(layers_to_plot):
        if layer_idx not in llm_attentions:
            continue

        attn = llm_attentions[layer_idx][0]  # [heads, seq, seq]

        # Average over heads
        attn_avg = attn.mean(dim=0).cpu().numpy()  # [seq, seq]

        # Top row: Full attention matrix (truncated for visibility)
        max_seq = min(50, attn_avg.shape[0])  # Show first 50 tokens
        im1 = axes[0, col].imshow(attn_avg[:max_seq, :max_seq], cmap='Blues', aspect='auto')
        axes[0, col].axvline(x=prefix_length-0.5, color='red', linestyle='--', linewidth=2, label='ECG prefix')
        axes[0, col].axhline(y=prefix_length-0.5, color='red', linestyle='--', linewidth=2)
        axes[0, col].set_title(f'Layer {layer_idx} Attention')
        axes[0, col].set_xlabel('Key position')
        axes[0, col].set_ylabel('Query position')
        plt.colorbar(im1, ax=axes[0, col])

        # Bottom row: How much non-prefix tokens attend to prefix
        # Average attention to prefix tokens from positions after prefix
        seq_len = attn_avg.shape[0]
        attn_to_prefix = attn_avg[prefix_length:, :prefix_length].mean(axis=0)

        axes[1, col].bar(range(prefix_length), attn_to_prefix, color='coral')
        axes[1, col].set_xticks(range(prefix_length))
        axes[1, col].set_xticklabels([f'ECG P{i}' for i in range(prefix_length)])
        axes[1, col].set_ylabel('Avg Attention')
        axes[1, col].set_title(f'Layer {layer_idx}: Text → ECG Prefix')

    plt.suptitle('LLaMA Attention to ECG Prefix Tokens', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved LLM attention to: {save_path}")

    plt.close(fig)
    return fig

def visualize_ecg_encoder_attention(encoder_attentions: Dict[int, torch.Tensor],
                                     ecg_signal: torch.Tensor,
                                     lead_idx: int = 1,
                                     fs: int = 500,
                                     layers_to_plot: Optional[List[int]] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize ECG encoder self-attention patterns.

    Shows which temporal positions in the ECG attend to which other positions.

    Args:
        encoder_attentions: Dict mapping layer_idx -> attention tensor
        ecg_signal: [12, 5000] ECG signal
        lead_idx: Which lead to display
        fs: Sampling frequency
        layers_to_plot: Which layers to visualize
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if not encoder_attentions:
        print("No encoder attention weights available")
        return None

    available_layers = sorted(encoder_attentions.keys())

    if layers_to_plot is None:
        layers_to_plot = [available_layers[0], available_layers[-1]]

    n_plots = len(layers_to_plot) + 1  # +1 for ECG signal
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3*n_plots))

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Plot ECG signal
    ecg = ecg_signal[lead_idx].cpu().numpy()
    time = np.arange(len(ecg)) / fs
    axes[0].plot(time, ecg, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'ECG Lead {lead_names[lead_idx]}')
    axes[0].grid(True, alpha=0.3)

    # Plot attention patterns for each layer
    for i, layer_idx in enumerate(layers_to_plot):
        if layer_idx not in encoder_attentions:
            continue

        attn = encoder_attentions[layer_idx]
        if attn.dim() == 4:
            attn = attn[0].mean(dim=0)  # [seq, seq]

        attn_np = attn.cpu().numpy()

        # Show attention matrix
        im = axes[i+1].imshow(attn_np, cmap='Blues', aspect='auto',
                              extent=[0, time[-1], time[-1], 0])
        axes[i+1].set_xlabel('Key Time (s)')
        axes[i+1].set_ylabel('Query Time (s)')
        axes[i+1].set_title(f'ECG Encoder Layer {layer_idx} Self-Attention')
        plt.colorbar(im, ax=axes[i+1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved encoder attention to: {save_path}")

    plt.close(fig)
    return fig

def visualize_combined_analysis(ecg_signal: torch.Tensor,
                                 saliency_dict: Dict[str, torch.Tensor],
                                 mapper_attention: torch.Tensor,
                                 llm_attentions: Dict[int, torch.Tensor],
                                 prefix_length: int = 4,
                                 lead_idx: int = 1,
                                 fs: int = 500,
                                 question: Optional[str] = None,
                                 answer: Optional[str] = None,
                                 prediction: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive visualization combining saliency and attention.

    Args:
        ecg_signal: [12, 5000] ECG tensor
        saliency_dict: Dict with 'encoder_only' and 'encoder_mapper' saliency
        mapper_attention: [batch, heads, seq, seq] mapper attention
        llm_attentions: Dict of LLM layer attentions
        prefix_length: Number of prefix tokens
        lead_idx: ECG lead to display
        fs: Sampling frequency
        question: Optional question text
        answer: Optional ground truth answer text
        prediction: Optional model prediction text
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(20, 16))

    # Create grid: 4 rows, 3 columns
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2], hspace=0.3, wspace=0.3)

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    ecg = ecg_signal[lead_idx].cpu().numpy()
    time = np.arange(len(ecg)) / fs

    # Row 1: ECG Signal (spans all columns)
    ax_ecg = fig.add_subplot(gs[0, :])
    ax_ecg.plot(time, ecg, 'b-', linewidth=0.8)
    ax_ecg.set_ylabel('Amplitude (mV)')
    ax_ecg.set_title(f'ECG Lead {lead_names[lead_idx]}', fontsize=12)
    ax_ecg.grid(True, alpha=0.3)

    # Row 2: Saliency maps
    ax_sal1 = fig.add_subplot(gs[1, 0])
    ax_sal2 = fig.add_subplot(gs[1, 1])
    ax_sal3 = fig.add_subplot(gs[1, 2])

    sal_enc = saliency_dict['encoder_only'].numpy()
    sal_enc_smooth = gaussian_filter1d(sal_enc, sigma=50)
    sal_enc_norm = (sal_enc_smooth - sal_enc_smooth.min()) / (sal_enc_smooth.max() - sal_enc_smooth.min() + 1e-8)

    sal_map = saliency_dict['encoder_mapper'].numpy()
    sal_map_smooth = gaussian_filter1d(sal_map, sigma=50)
    sal_map_norm = (sal_map_smooth - sal_map_smooth.min()) / (sal_map_smooth.max() - sal_map_smooth.min() + 1e-8)

    ax_sal1.fill_between(time, 0, sal_enc_norm, alpha=0.7, color='blue')
    ax_sal1.set_title('Encoder Saliency', fontsize=10)
    ax_sal1.set_ylabel('Importance')
    ax_sal1.set_xlabel('Time (s)')

    ax_sal2.fill_between(time, 0, sal_map_norm, alpha=0.7, color='red')
    ax_sal2.set_title('Encoder+Mapper Saliency', fontsize=10)
    ax_sal2.set_xlabel('Time (s)')

    ax_sal3.fill_between(time, 0, sal_enc_norm, alpha=0.5, color='blue', label='Encoder')
    ax_sal3.fill_between(time, 0, sal_map_norm, alpha=0.5, color='red', label='Mapper')
    ax_sal3.set_title('Comparison', fontsize=10)
    ax_sal3.set_xlabel('Time (s)')
    ax_sal3.legend(fontsize=8)

    # Row 3: Mapper Attention
    ax_mapper1 = fig.add_subplot(gs[2, 0])
    ax_mapper2 = fig.add_subplot(gs[2, 1:])

    attn_mapper = mapper_attention[0].mean(dim=0).cpu().numpy()
    seq_len_mapper = attn_mapper.shape[0]
    labels_mapper = [f'P{i}' for i in range(prefix_length)] + ['ECG']

    im_mapper = ax_mapper1.imshow(attn_mapper, cmap='Blues', aspect='auto')
    ax_mapper1.set_xticks(range(seq_len_mapper))
    ax_mapper1.set_yticks(range(seq_len_mapper))
    ax_mapper1.set_xticklabels(labels_mapper, fontsize=8)
    ax_mapper1.set_yticklabels(labels_mapper, fontsize=8)
    ax_mapper1.set_title('Mapper Attention', fontsize=10)
    plt.colorbar(im_mapper, ax=ax_mapper1)

    # Prefix attention to ECG
    ecg_attn = attn_mapper[:prefix_length, -1]
    ax_mapper2.bar(range(prefix_length), ecg_attn, color='steelblue', width=0.6)
    ax_mapper2.set_xticks(range(prefix_length))
    ax_mapper2.set_xticklabels([f'Prefix {i}' for i in range(prefix_length)])
    ax_mapper2.set_ylabel('Attention to ECG')
    ax_mapper2.set_title('How Prefix Tokens Attend to ECG Feature', fontsize=10)
    ax_mapper2.set_ylim(0, 1)

    # Row 4: LLM Attention to Prefix
    if llm_attentions:
        available_layers = sorted(llm_attentions.keys())
        n_layers = min(3, len(available_layers))
        layer_indices = [available_layers[0],
                        available_layers[len(available_layers)//2],
                        available_layers[-1]][:n_layers]

        for col, layer_idx in enumerate(layer_indices):
            ax_llm = fig.add_subplot(gs[3, col])

            attn_llm = llm_attentions[layer_idx][0].mean(dim=0).cpu().numpy()

            # Attention from text tokens to prefix
            if attn_llm.shape[0] > prefix_length:
                attn_to_prefix = attn_llm[prefix_length:, :prefix_length].mean(axis=0)
                ax_llm.bar(range(prefix_length), attn_to_prefix, color='coral')
                ax_llm.set_xticks(range(prefix_length))
                ax_llm.set_xticklabels([f'ECG P{i}' for i in range(prefix_length)], fontsize=8)
                ax_llm.set_ylabel('Avg Attention')
                ax_llm.set_title(f'LLaMA Layer {layer_idx}: Text→ECG', fontsize=10)

    # Add title with Q&A if provided
    title = 'ECG-QA Model Explainability Analysis'
    if question:
        title += f'\nQ: {question}'
    if answer:
        title += f'\nGround Truth: {answer}'
    if prediction:
        title += f' | Prediction: {prediction}'
    fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined analysis to: {save_path}")

    plt.close(fig)
    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================
def run_explainability_on_sample(meta_trainer, ecg_input, question_tokens, question_mask,
                                  answer_mask, sample_idx, output_dir, tokenizer,
                                  question_text=None, answer_text=None, ecg_id=None):
    """
    Run explainability analysis on a single sample.

    Args:
        meta_trainer: MetaTrainer instance with loaded model
        ecg_input: [1, 12, 5000] ECG tensor
        question_tokens: [1, seq_len] question tokens
        question_mask: [1, seq_len] question mask
        answer_mask: [1, seq_len] answer mask
        sample_idx: Index of this sample (for file naming)
        output_dir: Directory to save outputs
        tokenizer: Tokenizer for decoding predictions
        question_text: Optional question string
        answer_text: Optional answer string
        ecg_id: Optional ECG ID

    Returns:
        Dict with analysis results
    """
    model = meta_trainer.model
    model.eval()

    results = {}
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing Sample {sample_idx}" + (f" (ECG ID: {ecg_id})" if ecg_id else ""))
    print(f"{'='*60}")

    if question_text:
        print(f"Question: {question_text}")
    if answer_text:
        print(f"Ground Truth: {answer_text}")

    # Get model prediction
    fast_weights = list(model.mapper_net.parameters())
    with torch.no_grad():
        _, pred_tokens = model(
            ecg_input, question_tokens, question_mask, answer_mask,
            fast_weights, get_pred_tokens=True
        )
        prediction = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        print(f"Prediction: {prediction}")
        results['prediction'] = prediction

    # 1. Compute gradient-based saliency
    print("\n[1/4] Computing gradient-based saliency...")
    try:
        saliency_dict = compute_component_saliency(model, ecg_input)
        results['saliency'] = saliency_dict
    except Exception as e:
        print(f"Warning: Could not compute saliency: {e}")
        saliency_dict = None

    # 2. Extract attention weights
    print("[2/4] Extracting attention weights...")
    extractor = AttentionExtractor(model)

    # Mapper attention
    print("  - Mapper attention...")
    try:
        mapper_attn = extractor.extract_mapper_attention(ecg_input, fast_weights)
        results['mapper_attention'] = mapper_attn
    except Exception as e:
        print(f"Warning: Could not extract mapper attention: {e}")
        mapper_attn = None

    # LLM attention
    print("  - LLM attention...")
    try:
        llm_attn = extractor.extract_llm_attention(
            ecg_input, question_tokens, question_mask, answer_mask, fast_weights
        )
        results['llm_attention'] = llm_attn
    except Exception as e:
        print(f"Warning: Could not extract LLM attention: {e}")
        llm_attn = {}

    # ECG encoder attention
    print("  - ECG encoder attention...")
    try:
        encoder_attn = extractor.extract_ecg_encoder_attention(ecg_input)
        results['encoder_attention'] = encoder_attn
    except Exception as e:
        print(f"Warning: Could not extract encoder attention: {e}")
        encoder_attn = {}

    # 3. Generate visualizations
    print("[3/4] Generating visualizations...")

    ecg_signal = ecg_input.squeeze(0)
    prefix_length = model.prefix_length

    # Combined visualization
    if saliency_dict and mapper_attn is not None:
        visualize_combined_analysis(
            ecg_signal, saliency_dict, mapper_attn, llm_attn,
            prefix_length=prefix_length,
            question=question_text,
            answer=answer_text,
            prediction=prediction,
            save_path=os.path.join(sample_dir, 'combined_analysis.png')
        )

    # Separate visualizations
    if saliency_dict:
        visualize_component_comparison(
            ecg_signal, saliency_dict,
            question=question_text, answer=answer_text,
            save_path=os.path.join(sample_dir, 'saliency_comparison.png')
        )

    if mapper_attn is not None:
        visualize_mapper_attention(
            mapper_attn, prefix_length,
            save_path=os.path.join(sample_dir, 'mapper_attention.png')
        )

    if llm_attn:
        visualize_llm_attention_to_prefix(
            llm_attn, prefix_length,
            save_path=os.path.join(sample_dir, 'llm_attention.png')
        )

    if encoder_attn:
        visualize_ecg_encoder_attention(
            encoder_attn, ecg_signal,
            save_path=os.path.join(sample_dir, 'encoder_attention.png')
        )

    print(f"[4/4] Saved visualizations to: {sample_dir}")

    return result

def main_explain():
    """Main function for explainability analysis."""

    # Load tokenizer
    try:
        gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except AttributeError:
        print(f"Warning: Could not load tokenizer from {args.model_name}, trying direct HF load...")
        gpt_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

    experiment_id = "{}_{}-way_{}-shot{}".format(args.experiment_id, args.n_way, args.k_spt, args.model_type)

    # Load the trained model
    print(f"\n{'='*60}")
    print(f"Loading trained model: {experiment_id}")
    print(f"{'='*60}")

    meta_trainer = MetaTrainer(args, experiment_id, is_pretrained=True).to(device)

    # Freeze parameters for inference
    for param in meta_trainer.model.feature_extract.parameters():
        param.requires_grad = False
    for param in meta_trainer.model.gpt.parameters():
        param.requires_grad = False

    print(f"Loaded model from: {args.models_path}/{experiment_id}.pt")

    # Create output directory
    output_dir = os.path.join(args.output_dir, experiment_id, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Prepare data
    class_qa, train_temp, test_temp = prepare_ecg_qa_data(args)

    data_loader_test = FSL_ECG_QA_DataLoader(
        mode='test', n_way=args.n_way, k_shot=args.k_spt,
        k_query=args.k_qry, batchsz=args.batchsz_test,
        seq_len=args.seq_len, seq_len_a=args.seq_len_a,
        repeats=args.repeats, tokenizer=gpt_tokenizer,
        prefix_length=args.prefix_length, all_ids=class_qa,
        in_templates=test_temp, prompt=args.prompt,
        paraphrased_path=args.paraphrased_path,
        test_dataset=args.test_dataset, ecg_data_path=args.ecg_data_path
    )

    db_test = DataLoader(data_loader_test, batch_size=1, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    print(f"\n{'='*60}")
    print(f"Starting Explainability Analysis")
    print(f"Analyzing {args.num_samples} samples")
    print(f"{'='*60}")

    all_results = []

    for step, batch in enumerate(db_test):
        if step >= args.num_samples:
            break

        (
            x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a, id_spt,
            x_qry, y_qry_q, y_qry_a, y_qry_mask_q, y_qry_mask_a, qry_img_id
        ) = batch

        # Move to device
        x_spt = x_spt.to(device)
        y_spt_q = y_spt_q.to(device)
        y_spt_mask_q = y_spt_mask_q.to(device)
        y_spt_mask_a = y_spt_mask_a.to(device)

        # First do inner loop adaptation on support set
        fast_weights = meta_trainer.inner_loop_adaptation(
            x_spt[0], y_spt_q[0], y_spt_mask_q[0], y_spt_mask_a[0]
        )

        # Analyze query samples
        x_qry = x_qry.to(device)
        y_qry_q = y_qry_q.to(device)
        y_qry_mask_q = y_qry_mask_q.to(device)
        y_qry_mask_a = y_qry_mask_a.to(device)

        # Analyze first query sample from this task
        for qry_idx in range(min(1, x_qry.shape[1])):  # Just first query per task
            ecg_input = x_qry[0, qry_idx:qry_idx+1]  # [1, 12, 5000]
            q_tokens = y_qry_q[0, qry_idx:qry_idx+1]
            q_mask = y_qry_mask_q[0, qry_idx:qry_idx+1]
            a_mask = y_qry_mask_a[0, qry_idx:qry_idx+1]

            # Decode question and answer for display
            question_text = gpt_tokenizer.decode(q_tokens[0][q_mask[0] == 1], skip_special_tokens=True)
            answer_text = gpt_tokenizer.decode(y_qry_a[0, qry_idx][y_qry_mask_a[0, qry_idx] == 1], skip_special_tokens=True)
            ecg_id = qry_img_id[0][qry_idx] if qry_img_id else None

            # Update model with adapted weights for this task
            # Store original weights
            orig_weights = [p.clone() for p in meta_trainer.model.mapper_net.parameters()]

            # Set adapted weights
            for p, fw in zip(meta_trainer.model.mapper_net.parameters(), fast_weights):
                p.data = fw.data

            results = run_explainability_on_sample(
                meta_trainer, ecg_input, q_tokens, q_mask, a_mask,
                sample_idx=step,
                output_dir=output_dir,
                tokenizer=gpt_tokenizer,
                question_text=question_text,
                answer_text=answer_text,
                ecg_id=ecg_id
            )

            # Restore original weights
            for p, ow in zip(meta_trainer.model.mapper_net.parameters(), orig_weights):
                p.data = ow.data

            all_results.append({
                'sample_idx': step,
                'ecg_id': ecg_id,
                'question': question_text,
                'ground_truth': answer_text,
                'prediction': results.get('prediction', '')
            })

    # Save summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Explainability Analysis Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Samples analyzed: {len(all_results)}\n\n")

        for r in all_results:
            f.write(f"\nSample {r['sample_idx']}:\n")
            f.write(f"  ECG ID: {r['ecg_id']}\n")
            f.write(f"  Question: {r['question']}\n")
            f.write(f"  Ground Truth: {r['ground_truth']}\n")
            f.write(f"  Prediction: {r['prediction']}\n")

    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='ECG-QA Model Explainability Analysis')

    # Experiment identification
    argparser.add_argument('--experiment_id', type=int, default=123456,
                          help='Experiment ID of the trained model')
    argparser.add_argument('--model_type', type=str, default="",
                          help='Model type suffix (e.g., "_ablation")')

    # Model paths
    argparser.add_argument('--model_name', type=str, default="/llm_checkpoint/llama3.1-8b",
                          help='Path to LLM model')
    argparser.add_argument('--models_path', type=str, default='',
                          help='Path to saved model checkpoints')
    argparser.add_argument('--ecg_encoder_checkpoint', type=str, default='',
                          help='Path to ECG encoder checkpoint')

    # Data paths
    argparser.add_argument('--paraphrased_path', type=str, default='/ecgqa/ptbxl/paraphrased/',
                          help='Path to paraphrased ECG-QA json files')
    argparser.add_argument('--ecg_data_path', type=str, default='',
                          help='Path to ECG datasets')
    argparser.add_argument('--test_dataset', type=str, default="ptb-xl",
                          choices=["ptb-xl", "mimic"],
                          help='Dataset to use')

    # Output
    argparser.add_argument('--output_dir', type=str, default='./explainability_results',
                          help='Directory to save explainability outputs')
    argparser.add_argument('--num_samples', type=int, default=5,
                          help='Number of samples to analyze')

    # FSL parameters (must match training)
    argparser.add_argument('--n_way', type=int, default=5, help='N-way')
    argparser.add_argument('--k_spt', type=int, default=5, help='K-shot support')
    argparser.add_argument('--k_qry', type=int, default=5, help='K-shot query')
    argparser.add_argument('--batchsz_test', type=int, default=10)
    argparser.add_argument('--task_num', type=int, default=1)

    # Model parameters (must match training)
    argparser.add_argument('--seq_len', type=int, default=30)
    argparser.add_argument('--seq_len_a', type=int, default=30)
    argparser.add_argument('--prefix_length', type=int, default=4)
    argparser.add_argument('--mapper_type', type=str, default="ATT")
    argparser.add_argument('--prompt', type=int, default=1)

    # Training parameters (needed for MetaTrainer init)
    argparser.add_argument('--update_lr', type=float, default=0.05)
    argparser.add_argument('--meta_lr', type=float, default=5e-4)
    argparser.add_argument('--update_step', type=int, default=15)
    argparser.add_argument('--update_step_test', type=int, default=15)

    # Other
    argparser.add_argument('--question_type', type=str, default='single-verify')
    argparser.add_argument('--dif_exp', type=int, default=0)
    argparser.add_argument('--frozen_gpt', type=int, default=1)
    argparser.add_argument('--frozen_features', type=int, default=1)
    argparser.add_argument('--repeats', type=int, default=0)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--logs_path', type=str, default='')

    args = argparser.parse_args()

    if torch.cuda.is_available():
        print('Running explainability on GPU!')
    else:
        print('Running explainability on CPU!')

    main_explain()
