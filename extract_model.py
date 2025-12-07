from omegaconf import OmegaConf
from fairseq_signals import tasks
import torch
import torch.nn as nn


class ECGEncoderWrapper(nn.Module):
    """Wrapper to provide consistent output interface for different ECG encoders."""
    
    def __init__(self, model, model_type='baseline'):
        super().__init__()
        self.model = model
        self.model_type = model_type
    
    def forward(self, **kwargs):
        result = self.model(**kwargs)
        
        if self.model_type == 'ecg_fm':
            # wav2vec2_cmsc returns:
            #   'features': [batch, seq_len, 768] - this is what we need
            #   'x': [seq_len, batch, dim] - different format
            # Map to 'encoder_out' for compatibility with meta_learner.py
            if 'features' in result:
                result['encoder_out'] = result['features']
            if 'out' not in result:
                result['out'] = result.get('encoder_out', result.get('features'))
        
        return result


def hydra_main(checkpoint_path: str = ""):
    """
    Initialize and return the ECG feature extraction model.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint. 
                        Empty string (default) = random weights (baseline)
                        Path to .pt file = load pretrained ECG-FM weights
    
    Returns:
        model: ECG encoder model (wrapped for consistent interface)
    """
    
    use_pretrained = bool(checkpoint_path)
    
    if use_pretrained:
        # ECG-FM Pretrained config (W2V+CMSC, 12 layers, 768 dim)
        print(f"[ECG Encoder] Loading ECG-FM pretrained weights from: {checkpoint_path}")
        cfg = {
            'common': {
                'fp16': False,
                'log_format': 'csv',
                'log_interval': 10,
                'all_gather_list_size': 16384
            },
            'task': {
                '_name': 'ecg_pretraining',
                'data': '/content/data/process_ptbxl2_fsl',
                'normalize': False,
                'enable_padding': True,
                'enable_padding_leads': False,
                'leads_to_load': None,
                'perturbation_mode': [],
                'p': [],
                'mask_leads_selection': 'random',
                'mask_leads_prob': 0.0,
            },
            'dataset': {
                'num_workers': 4,
                'batch_size': 32,
                'valid_subset': 'valid'
            },
            'distributed_training': {
                'distributed_world_size': 1,
                'find_unused_parameters': False
            },
            'model': {
                '_name': 'wav2vec2_cmsc',
                'encoder_layers': 12,
                'encoder_embed_dim': 768,
                'encoder_ffn_embed_dim': 3072,
                'encoder_attention_heads': 12,
                'in_d': 12,
                'quantize_targets': True,
                'final_dim': 256,
                'dropout_input': 0.1,
                'dropout_features': 0.1,
                'feature_grad_mult': 0.1,
                'apply_mask': False,
                'mask_prob': 0.0,
            },
            'criterion': {
                '_name': 'wav2vec2_with_cmsc',
                'infonce': True,
                'log_keys': ['prob_perplexity', 'code_perplexity', 'temp'],
                'loss_weights': [0.1, 10]
            },
            'optimization': {
                'max_epoch': 200,
                'lr': [5e-5]
            },
            'optimizer': {
                '_name': 'adam',
                'adam_betas': '(0.9, 0.98)',
                'adam_eps': 1e-06,
                'weight_decay': 0.01
            },
            'lr_scheduler': {
                '_name': 'fixed',
                'warmup_updates': 0
            },
            'checkpoint': {
                'save_dir': 'checkpoints',
                'save_interval': 10
            }
        }
        model_type = 'ecg_fm'
    else:
        # Baseline config with random weights
        print("[ECG Encoder] Using random weights (baseline)")
        cfg = {
            'common': {
                'fp16': False,
                'log_format': 'json',
                'log_interval': 10,
                'all_gather_list_size': 2048000
            },
            'checkpoint': {
                'save_dir': 'checkpoints',
                'save_interval': 1,
                'keep_last_epochs': 1,
                'save_interval_updates': 0,
                'no_epoch_checkpoints': False
            },
            'task': {
                '_name': 'ecg_classification',
                'data': '/home/jtang/data/ecg_qa_500/ecg_qa/manifest/finetune',
                'path_dataset': False,
                'load_specific_lead': False
            },
            'dataset': {
                'num_workers': 6,
                'max_tokens': None,
                'batch_size': 128,
                'valid_subset': 'valid,test'
            },
            'distributed_training': {
                'distributed_world_size': 1
            },
            'criterion': {
                '_name': 'binary_cross_entropy_with_logits',
                'weight': None,
                'threshold': 0.5,
                'report_auc': True,
                'auc_average': 'macro',
                'pos_weight': None,
                'report_cinc_score': True,
                'weights_file': '/home/jtang/data/ecg_qa_500/ecg_qa/fairseq-signals/examples/w2v_cmsc/weights.csv',
                'per_log_keys': []
            },
            'optimization': {
                'max_epoch': 100,
                'max_update': 320000,
                'lr': [5e-05]
            },
            'optimizer': {
                '_name': 'adam',
                'adam_betas': '(0.9, 0.98)',
                'adam_eps': 1e-08,
                'weight_decay': 0.0,
                'use_old_adam': False,
                'lr': [5e-05]
            },
            'lr_scheduler': {
                '_name': 'fixed',
                'force_anneal': None,
                'lr_shrink': 0.1,
                'warmup_updates': 0,
                'lr': [5e-05]
            },
            'model': {
                '_name': 'ecg_transformer_classifier',
                'all_gather': False,
                'normalize': False,
                'filter': False,
                'data': '/home/jtang/data/ecg_qa_500/ecg_qa/manifest/finetune',
                'args': None,
                'encoder_layers': 12,
                'encoder_embed_dim': 768,
                'encoder_ffn_embed_dim': 3072,
                'encoder_attention_heads': 12,
                'layer_norm_first': False,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'activation_dropout': 0.1,
                'encoder_layerdrop': 0.0,
                'dropout_input': 0.0,
                'dropout_features': 0.0,
                'apply_mask': False,
                'mask_length': 10,
                'mask_prob': 0.0,
                'mask_selection': 'static',
                'mask_other': 0.0,
                'no_mask_overlap': False,
                'mask_min_space': 1,
                'mask_channel_length': 10,
                'mask_channel_prob': 0.0,
                'mask_channel_selection': 'static',
                'mask_channel_other': 0.0,
                'no_mask_channel_overlap': False,
                'mask_channel_min_space': 1,
                'extractor_mode': 'default',
                'conv_feature_layers': '[(256, 2, 2)] * 4',
                'in_d': 12,
                'conv_bias': False,
                'feature_grad_mult': 0.0,
                'conv_pos': 128,
                'conv_pos_groups': 16,
                'model_path': None,
                'no_pretrained_weights': True,
                'freeze_finetune_updates': 0,
                'final_dropout': 0.0,
                'num_labels': 26
            },
            'job_logging_cfg': {
                'version': 1,
                'formatters': {'simple': {'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'}},
                'handlers': {
                    'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'},
                    'file': {'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'extract.log'}
                },
                'root': {'level': 'INFO', 'handlers': ['console', 'file']},
                'disable_existing_loggers': False
            }
        }
        model_type = 'baseline'
    
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    
    # Manually load pretrained weights for ECG-FM
    if use_pretrained:
        print(f"  - Loading checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  - Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Wrap model for consistent interface
    wrapped_model = ECGEncoderWrapper(model, model_type=model_type)
    
    return wrapped_model

if __name__ == "__main__":
    model = hydra_main()
    model.cuda()

    # Prepare padding mask and input tensor
    # Generate a random test ECG input
    truncated_ecg = torch.randn(1, 12, 5000)  # batch_size=1, leads=12, length=5000
    padding_mask = torch.zeros(truncated_ecg.shape[0], 12, 5000, dtype=torch.bool).cuda()
    inp = truncated_ecg.cuda()

    # Pass the test ECG through the model
    proj_out = model(**{
        'source': inp,
        'padding_mask': padding_mask
    })
    print("Model output shape:", proj_out['out'].shape)


