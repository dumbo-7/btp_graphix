#!/usr/bin/env python
import os
import json
import shutil
import argparse
import configparser
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, AutoConfig
from tqdm import tqdm

from models.graphix.rgat import Model
from discourse_graph.utils import prepare_data, decode_anl
from discourse_graph.evaluate import BatchEvaluator

# ------------------------- CLI / config ------------------ #
# Read configurations from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Argument parsing setup
parser = argparse.ArgumentParser(description="Script for training and testing a model")

# Add arguments for each configuration value in config.ini
parser.add_argument('--do_train', type=bool, default=None, help="Whether to perform training")
parser.add_argument('--do_test', type=bool, default=None, help="Whether to perform testing")
parser.add_argument('--num_steps', type=int, default=None, help="Number of training steps")
parser.add_argument('--num_runs', type=int, default=None, help="Number of training runs")
parser.add_argument('--step_interval', type=int, default=None, help="Interval between checkpoint saves")
parser.add_argument('--learning_rate', type=float, default=None, help="Learning rate for optimizer")
parser.add_argument('--max_seq_length', type=int, default=None, help="Maximum sequence length")
parser.add_argument('--batch_size', type=int, default=None, help="Batch size for training")
parser.add_argument('--batch_size_inference', type=int, default=None, help="Batch size for inference")
parser.add_argument('--model_name_or_path', type=str, default=None, help="Pretrained model name or path")
parser.add_argument('--dataset_name', type=str, default=None, help="Name of the dataset to use")
parser.add_argument('--train_on_both', type=bool, default=None, help="Whether to train on both train and dev sets")
parser.add_argument('--variant_name', type=str, default=None, help="Variant name")
parser.add_argument('--resume_from_checkpoint', type=bool, default=None, help="Whether to resume from checkpoint")
parser.add_argument('--test_at_checkpoint', type=bool, default=None, help="Whether to test at each checkpoint")

args = parser.parse_args()

# Overwrite config values with command-line arguments, if provided
for key in vars(args):
    value = getattr(args, key)
    if value is not None:
        config.set('DEFAULT', key, str(value))

# Extract configuration values
do_train = config.getboolean('DEFAULT', 'do_train')
do_test = config.getboolean('DEFAULT', 'do_test')
num_steps = config.getint('DEFAULT', 'num_steps')
num_runs = config.getint('DEFAULT', 'num_runs')
step_interval = config.getint('DEFAULT', 'step_interval')
learning_rate = config.getfloat('DEFAULT', 'learning_rate')
max_seq_length = config.getint('DEFAULT', 'max_seq_length')
max_seq_length_inference = config.getint('DEFAULT', 'max_seq_length_inference')
batch_size = config.getint('DEFAULT', 'batch_size')
batch_size_inference = config.getint('DEFAULT', 'batch_size_inference')
model_name_or_path = config.get('DEFAULT', 'model_name_or_path')
dataset_name = config.get('DEFAULT', 'dataset_name')
train_on_both = config.getboolean('DEFAULT', 'train_on_both')
variant_name = config.get('DEFAULT', 'variant_name')
resume_from_checkpoint = config.getboolean('DEFAULT', 'resume_from_checkpoint')
test_at_checkpoint = config.getboolean('DEFAULT', 'test_at_checkpoint')


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load discourse graphs
graph_pedia = pickle.load(open("./discourse_graph/graph_pedia_discourse.bin", "rb"))

# tokenizer
tokenizer = T5TokenizerFast.from_pretrained(
    model_name_or_path,
    model_max_length=max_seq_length,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# dataset class
class CustomDataset(Dataset):
    def __init__(self, enc_in, enc_tgt):
        self.enc_in  = enc_in
        self.enc_tgt = enc_tgt
    def __len__(self): return len(self.enc_in["input_ids"])
    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc_in ["input_ids"][idx],
            "attention_mask": self.enc_in ["attention_mask"][idx],
            "labels":         self.enc_tgt["input_ids"][idx],
            "graph_idx":      torch.tensor(idx, dtype=torch.long), #changed idx to tensor, previously it was idx
        }



# @torch.no_grad()
# def perform_inference(model, dl, phase_name="Evaluation"):
#     model.eval()
#     evaluator = BatchEvaluator()
#     pbar = tqdm(dl, desc=phase_name, unit="batch")
#     for batch in pbar:
#         # Move all inputs to device
        
#         # batch = {k: v.to(device) for k, v in batch.items()}
#         batch = {
#             k: (v.to(device) if isinstance(v, torch.Tensor) else v)
#             for k, v in batch.items()
#         }

#         # Forward pass (graph_idx is passed through kwargs)
#         out = model(**batch)
#         loss = out.get('loss')
#         pbar.set_postfix({'loss': f"{loss:.4f}"})

#         # Generation with discourse graphs
#         gen_ids = model.generate(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             max_length=max_seq_length_inference,
#             graph_idx=batch["graph_idx"],
#         )
#         preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
#         labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

#         # Print examples
#         for t, p in zip(labels, preds):
#             tqdm.write(f"True:{t}\nPred:{p}\n{'-'*80}\n")

#         # Evaluation metrics
#         proc = [decode_anl(p) for p in preds]
#         truth_proc = [decode_anl(t) for t in labels]
#         evaluator.add_batch(
#             [x[0] for x in truth_proc], [x[1] for x in truth_proc],
#             [x[0] for x in proc],        [x[1] for x in proc]
#         )
#     pbar.close()
#     return evaluator.evaluate()

@torch.no_grad()
def perform_inference(model, dl, phase_name="Evaluation"):
    model.eval()
    evaluator = BatchEvaluator()
    pbar = tqdm(dl, desc=phase_name, unit="batch")
    
    for batch in pbar:
        # Move all inputs to device
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        # Forward pass (graph_idx is passed through kwargs)
        out = model(**batch)
        loss = out.get('loss')
        pbar.set_postfix({'loss': f"{loss:.4f}"})

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        graph_idx = batch["graph_idx"]
        
        # Build allowed token sets for each item in batch
        allowed_tokens_per_sample = [
            set(input_id.tolist()) for input_id in input_ids
        ]

        # Generation with decoding constraints
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_seq_length_inference,
            graph_idx=graph_idx,
            num_beams=4,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # Print examples
        for t, p in zip(labels, preds):
            tqdm.write(f"True:{t}\nPred:{p}\n{'-'*80}\n")

        # Evaluation metrics
        proc = [decode_anl(p) for p in preds]
        truth_proc = [decode_anl(t) for t in labels]
        evaluator.add_batch(
            [x[0] for x in truth_proc], [x[1] for x in truth_proc],
            [x[0] for x in proc],        [x[1] for x in proc]
        )

    pbar.close()
    return evaluator.evaluate()


def train_one_epoch(model, dl, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dl, desc="Training", unit="batch"):
        # Move inputs to device
        
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch = {
        #     k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        #     for k, v in batch.items()
        # }

        # Mask padding tokens in labels
        batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(**batch)
        loss = out.get('loss')
        if loss is None:
            raise ValueError("Model output does not contain 'loss'")

        # Backward & optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dl)



def main():
    # load raw data
    train_js = json.load(open(f"./discourse_graph/datasets/{dataset_name}/{dataset_name}_train.json"))
    dev_js   = json.load(open(f"./discourse_graph/datasets/{dataset_name}/{dataset_name}_dev.json"))
    test_js  = json.load(open(f"./discourse_graph/datasets/{dataset_name}/{dataset_name}_test.json"))

    # prepare text inputs and targets
    train_in, train_out = prepare_data(train_js)
    dev_in,   dev_out   = prepare_data(dev_js)
    test_in,  test_out  = prepare_data(test_js)
    if train_on_both:
        train_in  += dev_in
        train_out += dev_out
        
    def pack_ids(id_lists, pad_id, target_len=None):
        """Pad variable‑length ID lists to a uniform tensor batch."""
        max_len = target_len if target_len is not None else max(len(x) for x in id_lists)
        ids_mat = []
        attn_mat = []
        for ids in id_lists:
            pad = [pad_id] * (max_len - len(ids))
            ids_mat.append(torch.tensor(ids + pad))
            attn_mat.append(torch.tensor([1]*len(ids) + [0]*len(pad)))
        return {
            "input_ids": torch.stack(ids_mat),
            "attention_mask": torch.stack(attn_mat),
        }

    
    # Get node features from graph_pedia
    tr_ids = [graph_pedia[i]["features"].tolist() for i in range(len(train_in))]
    dv_ids = [graph_pedia[len(train_in)+i]["features"].tolist() for i in range(len(dev_in))]
    ts_ids = [graph_pedia[len(train_in)+len(dev_in)+i]["features"].tolist() for i in range(len(test_in))]

    # Manually compute max graph size (or use a fixed one like 250)
    MAX_GRAPH_LEN = max(len(x) for x in tr_ids + dv_ids + ts_ids)
    print(f"[INFO] Setting max graph feat length to {MAX_GRAPH_LEN}")

    # Encode using graph features
    tr_enc_in = pack_ids(tr_ids, pad_id=tokenizer.pad_token_id, target_len=MAX_GRAPH_LEN)
    dv_enc_in = pack_ids(dv_ids, pad_id=tokenizer.pad_token_id, target_len=MAX_GRAPH_LEN)
    ts_enc_in = pack_ids(ts_ids, pad_id=tokenizer.pad_token_id, target_len=MAX_GRAPH_LEN)


    # tokenize
    tr_enc_out = tokenizer(train_out, padding=True, truncation=True, return_tensors="pt")
    dv_enc_out = tokenizer(dev_out,   padding=True, truncation=True, return_tensors="pt")
    ts_enc_out = tokenizer(test_out,  padding=True, truncation=True, return_tensors="pt")

    # dataloaders
    train_dl = DataLoader(CustomDataset(tr_enc_in,  tr_enc_out), batch_size=batch_size, shuffle=True)
    dev_dl   = DataLoader(CustomDataset(dv_enc_in,  dv_enc_out), batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(CustomDataset(ts_enc_in,  ts_enc_out), batch_size=batch_size, shuffle=False)
    
    # build Graphix-T5 model
    from types import SimpleNamespace
    model_args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        cache_dir=None, model_revision=None,
        use_auth_token=False, launch_picard=False,
        use_picard=False
    )
    config = AutoConfig.from_pretrained(model_name_or_path)
    model  = Model(
        tokenizer, lambda cls: cls,
        model_args, config,
        graph_pedia
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1 - step / num_steps)
    )

    # run training + checkpointing
    best_score = -1.0
    out_dir = f"./results/{dataset_name}-{variant_name}-{model_name_or_path}" + \
              f"-steps{num_steps}-BS{batch_size}-Len{max_seq_length}"
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_dir = os.path.join(out_dir, "best_checkpoint")

# ------------------------------- EPOCHWISE  EVALUATION ----------------------------------------------------------------
    # if do_train:
    #     total_steps   = 0
    #     best_score    = -1.0
    #     best_ckpt_dir = os.path.join(out_dir, "best_checkpoint")

    #     steps_per_epoch = len(train_dl)
    #     while total_steps < num_steps:
    #         # ──────────────── 1. train one epoch ────────────────
    #         avg_loss = train_one_epoch(model, train_dl, optimizer, scheduler)
    #         total_steps += steps_per_epoch
    #         print(f"[epoch done] global_step={total_steps}  |  loss={avg_loss:.4f}")

    #         # ──────────────── 2. evaluate on dev ────────────────
    #         val_res = perform_inference(model, dev_dl, "Validation")
    #         comb_f1 = sum(val_res[k] for k in ["ACI_f1", "ACC_f1",
    #                                         "ARI_f1", "ARC_f1"])
    #         print(f"    → dev combined F1 = {comb_f1:.4f}")

    #         # ──────────────── 3. checkpoint logic ───────────────
    #         is_best = comb_f1 > best_score
    #         if is_best:
    #             best_score = comb_f1

    #             # (a) remove previous best (if any)
    #             if os.path.isdir(best_ckpt_dir):
    #                 shutil.rmtree(best_ckpt_dir)

    #             # (b) save new best
    #             os.makedirs(best_ckpt_dir, exist_ok=True)
    #             model.save_pretrained(best_ckpt_dir)
    #             tokenizer.save_pretrained(best_ckpt_dir)
    #             torch.save(optimizer.state_dict(),
    #                     os.path.join(best_ckpt_dir, "optimizer.pt"))
    #             torch.save(scheduler.state_dict(),
    #                     os.path.join(best_ckpt_dir, "scheduler.pt"))
    #             print(f"    ✓ new BEST checkpoint saved (F1={best_score:.4f})")
    #         else:
    #             print("    (no improvement)")

    #     # ─────────────────────── 4. final test ───────────────────────
    # if do_test:
    #     print(f"\nTesting best checkpoint at {best_ckpt_dir}\n")

    #     # 1) rebuild an empty wrapper
    #     cfg   = AutoConfig.from_pretrained(model_name_or_path)
    #     best_model = Model(
    #         tokenizer, lambda cls: cls,          # same ctor args as before
    #         model_args, cfg,
    #         graph_pedia
    #     ).to(device)

    #     # 2) load the saved weights
    #     state_path = os.path.join(best_ckpt_dir, "pytorch_model.bin")
    #     best_model.load_state_dict(
    #         torch.load(state_path, map_location=device),
    #         strict=False                           # ignore keys that belong only to T5
    #     )

    #     # 3) evaluate
    #     test_res = perform_inference(best_model, test_dl, "Final Testing")

    #     with open(os.path.join(best_ckpt_dir, "final_test_score.json"), "w") as fh:
    #         json.dump(test_res, fh, indent=4)

    #     print("Test Results:")
    #     for k, v in test_res.items():
    #         print(f"{k:20}: {v:.4f}")

# -------------------------------- STEPWISE  EVALUATION ------------------------------------------------------------

    if do_train:
        global_step   = 0
        best_score    = -1.0
        best_ckpt_dir = os.path.join(out_dir, "best_checkpoint")
        os.makedirs(out_dir, exist_ok=True)

        train_iter = iter(train_dl)          # infinite iterator over batches
        pbar = tqdm(total=num_steps, desc="Training", unit="step")

        while global_step < num_steps:
            # -------------------- 1. get next batch -----------------------
            try:
                batch = next(train_iter)
            except StopIteration:            # restart dataloader each epoch pass
                train_iter = iter(train_dl)
                batch = next(train_iter)

            # -------------------- 2. move + mask --------------------------
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100

            # -------------------- 3. forward / backward -------------------
            model.train()
            optimizer.zero_grad()
            out  = model(**batch)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # -------------------- 4. dev-eval & checkpoint --------------- 
            should_eval = (global_step % step_interval == 0) or (global_step == num_steps)
            if should_eval:
                val_res = perform_inference(model, dev_dl, f"Dev @ step {global_step}")
                comb_f1 = sum(val_res[k] for k in ["ACI_f1", "ACC_f1", "ARI_f1", "ARC_f1"])
                print(f"\n[step {global_step}] dev combined F1 = {comb_f1:.4f}")

                if comb_f1 > best_score:                  # new best → save
                    best_score = comb_f1
                    if os.path.isdir(best_ckpt_dir):
                        shutil.rmtree(best_ckpt_dir)
                    os.makedirs(best_ckpt_dir, exist_ok=True)

                    model.save_pretrained(best_ckpt_dir, safe_serialization=False)
                    tokenizer.save_pretrained(best_ckpt_dir)
                    torch.save(optimizer.state_dict(), os.path.join(best_ckpt_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(best_ckpt_dir, "scheduler.pt"))
                    print(f"    ✓ new BEST checkpoint saved  (F1={best_score:.4f})")
                    with open(os.path.join(best_ckpt_dir, f"val_results_{global_step}.json"), "w") as fh:
                        json.dump(val_res, fh, indent=4)
                else:
                    print("    (no improvement)")

        pbar.close()

    if do_test:
        print(f"\nTesting best checkpoint at {best_ckpt_dir}\n")

        # 1) rebuild an empty wrapper
        cfg   = AutoConfig.from_pretrained(model_name_or_path)
        best_model = Model(
            tokenizer, lambda cls: cls,          # same ctor args as before
            model_args, cfg,
            graph_pedia
        ).to(device)

        # 2) load the saved weights
        state_path = os.path.join(best_ckpt_dir, "pytorch_model.bin")
        best_model.load_state_dict(
            torch.load(state_path, map_location=device),
            strict=False                           # ignore keys that belong only to T5
        )

        # 3) evaluate
        test_res = perform_inference(best_model, test_dl, "Final Testing")

        with open(os.path.join(best_ckpt_dir, "best_checkpoint_test_score.json"), "w") as fh:
            json.dump(test_res, fh, indent=4)

        print("Test Results:")
        for k, v in test_res.items():
            print(f"{k:20}: {v:.4f}")



if __name__ == "__main__":
    main()
