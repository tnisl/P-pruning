# P-pruning
Official code for LREC-COLING2024 paper "Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models"[[paper link](https://aclanthology.org/2024.lrec-main.1162/)]
<div align="center">
  <img src=figure/overview.pdf>
</div>


# Finetuning Settings
The experimental settings for finetuning the pruned models are listed as below.
| Model |                        Task                        |   learning rate  | batch size |  epoch  |
|:-----:|:--------------------------------------------------:|:----------------:|:----------:|:-------:|
|  bert | `stsb`, `mrpc`, `sst2`, `qnli`, `qqp`, `mnli`, `squad`, `squad_v2` | 5e-5, 3e-5, 2e-5 |   16, 32   | 2, 3, 4 |
|  gpt2 |                           `penn_treebank`                          |       1e-4       |     32     |    10   |
|       |                           `wikitext-2-v1`                          |       5e-4       |     32     |    8    |
|       |                          `wikitext-103-v1`                         |       1e-4       |     32     |    10   |


# Selected Table 2 Reproduction
We reproduced the selected BERT-base conditions from Table 2 for MRPC, STS-B, SST-2, and SQuAD 1.1. The paper values are shown alongside our scores; deltas are percentage points (run minus paper).

| Task (metric) | BERTBASE | Random | +MC | +CS |
|:--|--:|--:|--:|--:|
| MRPC (accuracy) | 86.27 → 81.86 (−4.41) | 77.45 → 76.96 (−0.49) | 81.86 → 81.13 (−0.73) | 86.03 → 82.60 (−3.43) |
| STS-B (Spearman) | 88.59 → 87.49 (−1.10) | 85.82 → 82.51 (−3.31) | 86.70 → 84.67 (−2.03) | 88.12 → 86.29 (−1.83) |
| SST-2 (accuracy) | 93.12 → 91.97 (−1.15) | 90.83 → 91.40 (+0.57) | 91.74 → 92.43 (+0.69) | 92.32 → 91.40 (−0.92) |
| SQuAD 1.1 (F1) | 88.48 → 88.03 (−0.45) | 84.79 → 84.09 (−0.70) | 85.37 → 84.86 (−0.51) | 86.70 → 85.55 (−1.15) |

The gains over random pruning are:

| Task | Paper +MC | Reproduction +MC | Paper +CS | Reproduction +CS |
|:--|--:|--:|--:|--:|
| MRPC | +4.41 | +4.17 | +8.58 | +5.64 |
| STS-B | +0.88 | +2.16 | +2.30 | +3.79 |
| SST-2 | +0.91 | +1.03 | +1.49 | 0.00 |
| SQuAD 1.1 | +0.58 | +0.77 | +1.91 | +1.46 |

**Run settings:** `bert-base-uncased`, seed 3407, 3,000 calibration examples, 200 sampled tokens, λ=5, δ=0.001, and a 60% relative-FLOPs target. Fine-tuning used learning rate 2e-5 for 3 epochs, batch size 32 for GLUE tasks and 12 for SQuAD, with maximum sequence lengths 128 and 384 respectively. The run used a Tesla T4. BERTBASE is the fine-tuned unpruned model in our run; the paper score is a reference. The paper does not specify the exact task-specific fine-tuning recipe used for Table 2, so this is a single-seed reproduction with fixed settings. The SQuAD +CS condition was run separately and completed all 3 epochs; its Exact Match was 77.04 (not reported in Table 2).

For SQuAD, use the `raw_score` or `eval_metrics_raw` values in the generated run report. Its `metrics_percent` field is currently scaled by an extra factor of 100.

Full run reports: [initial run (all conditions except SQuAD +CS)](experiments/table2_selected_tasks/initial_run_report.json) and [SQuAD +CS completion](experiments/table2_selected_tasks/squad_cs_run_report.json).

# Stage 1: Prune models
The target model is first pruned with `./mask.py` to obtain the `head_mask.pt` and `neuron_mask.pt`. The masks are saved at `/PATH/TO/OUTPUT/$TASK`.

    cd /PATH/TO/P-pruning
    MODEL=bert
    TASK=stsb
    SAVE_DIR=/PATH/TO/OUTPUT/$TASK
    python ./mask.py \
        --model $MODEL \
        --save_dir $SAVE_DIR \
        --task $TASK \
        --max_instances 3000 \
        --pruning_ratio 0.3 \
        --sampling_tokens 200 \
        --lambda_val 5

  # Stage 2: Finetune models
  The pruned target model is then finetuned towards the downstream task with `./finetune.py`.

    python /GPFS/data/pingjiewang/P-pruning/finetune.py \
        --model_name_or_path $MODEL \
        --task_name $TASK_NAME \
        --mask_dir $MASK_DIR \
        --output_dir $OUTPUT_DIR \
        --seed 3407 \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 4.0

# Citation
```bibtex
@inproceedings{wang2024pruning,
              title={Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models},
              author={Wang, Pingjie and Liu, Hongcheng and Wang, Yanfeng and Wang, Yu},
              booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
              pages={13279--13289},
              year={2024}
            }
```
