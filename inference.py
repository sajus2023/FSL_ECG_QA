import argparse
import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from load_class import prepare_ecg_qa_data
from data_loader import FSL_ECG_QA_DataLoader
from meta_trainer import MetaTrainer
from utils import *
import numpy as np
import os

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

device = set_device()
if torch.cuda.is_available():
    print('Inference on GPU!')
else:
    print('Inference on CPU!')

def write_data_to_txt(file_path, data):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)

def main_inference():
    # Load tokenizer with error handling
    try:
        gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except AttributeError:
        print(f"Warning: Could not load tokenizer from {args.model_name}, trying direct HF load...")
        gpt_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

    experiment_id = "{}_{}-way_{}-shot{}".format(args.experiment_id, args.n_way, args.k_spt, args.model_type)

    # Load the trained model by setting is_pretrained=True
    meta_trainer = MetaTrainer(args, experiment_id, is_pretrained=True).to(device)
    
    if args.frozen_features == 1:
        for param in meta_trainer.model.feature_extract.parameters():
            param.requires_grad = False
    if args.frozen_gpt == 1:
        for param in meta_trainer.model.gpt.parameters():
            param.requires_grad = False
            
    params = list(filter(lambda p: p.requires_grad, meta_trainer.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
    print(f"Loaded trained model from: {args.models_path}/{experiment_id}.pt")

    log_file_path = args.logs_path + "/log_{}_inference.txt".format(experiment_id)
    write_data_to_txt(log_file_path, "Experiment ID: {} Date: {}, {}-way, {}-shot (support), {}-shot (query), Test Dataset: {}\n"
                      .format(experiment_id, datetime.datetime.now(), args.n_way, args.k_spt, args.k_qry, args.test_dataset))

    class_qa, train_temp, test_temp = prepare_ecg_qa_data(args)

    data_loader_test = FSL_ECG_QA_DataLoader(mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                     k_query=args.k_qry, batchsz=args.batchsz_test,
                                     seq_len=args.seq_len, seq_len_a=args.seq_len_a,
                                     repeats=args.repeats, tokenizer=gpt_tokenizer,
                                     prefix_length=args.prefix_length, all_ids=class_qa,
                                     in_templates=test_temp, prompt=args.prompt,
                                     paraphrased_path=args.paraphrased_path,
                                     test_dataset=args.test_dataset, ecg_data_path=args.ecg_data_path)
                                     
    db_test = DataLoader(data_loader_test, batch_size=args.task_num, shuffle=True, 
                        num_workers=args.num_workers, pin_memory=True)
                        
    print(f"\n{'='*60}")
    print(f"Starting Inference on {len(db_test)} test tasks")
    print(f"{'='*60}\n")

    accs_all_test = []

    for step, batch in enumerate(db_test):
        print(f"\n[Task {step+1}/{len(db_test)}]")

        (
            x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a, id_spt,
            x_qry, y_qry_q, y_qry_a, y_qry_mask_q, y_qry_mask_a, qry_img_id
        ) = batch

        x_spt        = x_spt.to(device)
        y_spt_q      = y_spt_q.to(device)
        y_spt_a      = y_spt_a.to(device)
        y_spt_mask_q = y_spt_mask_q.to(device)
        y_spt_mask_a = y_spt_mask_a.to(device)
        # id_spt stays on CPU
        x_qry        = x_qry.to(device)
        y_qry_q      = y_qry_q.to(device)
        y_qry_a      = y_qry_a.to(device)
        y_qry_mask_q = y_qry_mask_q.to(device)
        y_qry_mask_a = y_qry_mask_a.to(device)
        # qry_img_id stays on CPU

        # Call finetunning with calc_metrics=True and log_predictions=True for inference
        accs, bmr_metrics, bleu_metrics = meta_trainer.finetunning(
            x_spt, y_spt_q, y_spt_a, y_spt_mask_q, y_spt_mask_a,
            x_qry, y_qry_q, y_qry_mask_q, y_qry_mask_a, y_qry_a,
            calc_metrics=True, qry_ecg_ids=qry_img_id, log_predictions=True
        )

        accs_all_test.append((accs, bmr_metrics, bleu_metrics))

        print(f"[Task {step+1}/{len(db_test)}] Test acc: {accs[-1]:.4f}")
        write_data_to_txt(file_path=log_file_path,
                          data=f"Task {step+1}: Test acc: {accs}\n")

    # Calculate average accuracy across all test tasks
    acc_arrays = [item[0] for item in accs_all_test]
    accs_mean = np.array(acc_arrays).mean(axis=0).astype(np.float16)

    # Calculate mean metrics
    avg_bertscore = np.mean([item[1]['f1_bertscore'] for item in accs_all_test])
    avg_meteor = np.mean([item[1]['meteor'] for item in accs_all_test])
    avg_rouge = np.mean([item[1]['rouge'] for item in accs_all_test])

    avg_bleu1 = np.mean([item[2]['BLEU-1'] for item in accs_all_test])
    avg_bleu2 = np.mean([item[2]['BLEU-2'] for item in accs_all_test])
    avg_bleu3 = np.mean([item[2]['BLEU-3'] for item in accs_all_test])
    avg_bleu4 = np.mean([item[2]['BLEU-4'] for item in accs_all_test])

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accs_mean[-1]:.4f}")
    print(f"\nBERTScore Metrics:")
    print(f"  F1: {avg_bertscore:.4f}")
    print(f"  METEOR: {avg_meteor:.4f}")
    print(f"  ROUGE: {avg_rouge:.4f}")
    print(f"\nBLEU Metrics:")
    print(f"  BLEU-1: {avg_bleu1:.4f}")
    print(f"  BLEU-2: {avg_bleu2:.4f}")
    print(f"  BLEU-3: {avg_bleu3:.4f}")
    print(f"  BLEU-4: {avg_bleu4:.4f}")
    print(f"{'='*60}\n")

    metrics_str = (
        f"\nFINAL METRICS:\n"
        f"  Test Accuracy: {accs_mean[-1]:.4f}\n"
        f"  BERTScore F1: {avg_bertscore:.4f}\n"
        f"  METEOR: {avg_meteor:.4f}\n"
        f"  ROUGE: {avg_rouge:.4f}\n"
        f"  BLEU-1: {avg_bleu1:.4f}\n"
        f"  BLEU-2: {avg_bleu2:.4f}\n"
        f"  BLEU-3: {avg_bleu3:.4f}\n"
        f"  BLEU-4: {avg_bleu4:.4f}\n"
        f"All accuracies: {accs_mean}\n"
    )

    write_data_to_txt(file_path=log_file_path, data=metrics_str)
    write_data_to_txt(log_file_path, f"Experiment completed: {experiment_id} Date: {datetime.datetime.now()}\n")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment_id', type=int, default=123456)
    argparser.add_argument('--batchsz_test', type=int, default=10)
    argparser.add_argument('--paraphrased_path', type=str, default='/ecgqa/ptbxl/paraphrased/',
                          help='path to ./paraphrased containing train/val/test ECG-QA json files')
    argparser.add_argument('--test_dataset', type=str, default="ptb-xl", choices=["ptb-xl", "mimic"],
                          help='Dataset to use (ptb-xl or mimic)')
    argparser.add_argument('--model_type', type=str, help='model need to test', default="")
    argparser.add_argument('--model_name', type=str, help="path to llm model",
                          default="/llm_checkpoint/llama3.1-8b")
    argparser.add_argument('--ecg_data_path', type=str, help='the path to datasets', default='')
    argparser.add_argument('--question_type', type=str, default='single-verify',
                          help='question types: single-verify, single-choose, single-query, all')
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--prompt', type=int, default=1,
                          help='1=Question: +q_str+Answer:, 2=q_str, 3=q_str+answer options')
    argparser.add_argument('--dif_exp', type=int, help='0=same_exp, 1=dif_exp', default=0)
    argparser.add_argument('--frozen_gpt', type=int, help='0=unfrozen_gpt, 1=frozen_gpt', default=1)
    argparser.add_argument('--frozen_features', type=int, help='0=unfrozen_features, 1=frozen_features', default=1)
    argparser.add_argument('--repeats', type=int, help='repeats for support set', default=0)
    argparser.add_argument('--seq_len', type=int, default=30)
    argparser.add_argument('--seq_len_a', type=int, default=30)
    argparser.add_argument('--prefix_length', type=int, default=4)
    argparser.add_argument('--mapper_type', type=str, default="ATT", help='Type of mapper: MLP or ATT')
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=15)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning', default=15)
    argparser.add_argument('--models_path', type=str, default='', help='path to saved model checkpoints')
    argparser.add_argument('--logs_path', type=str, default='', help='path to save logs')
    argparser.add_argument('--ecg_encoder_checkpoint', type=str, default='', help='path to the ecg encoder checkpoint file')

    args = argparser.parse_args()

    main_inference()
