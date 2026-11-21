# P-pruning

Mã nguồn chính thức cho bài báo tại LREC-COLING 2024, “Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models” ([bài báo](https://aclanthology.org/2024.lrec-main.1162/)).

<div align="center">
  <img src=figure/overview.pdf>
</div>

## Cấu hình fine-tuning

| Mô hình | Tác vụ | Learning rate | Batch size | Epoch |
|:--:|:--|:--:|:--:|:--:|
| BERT | `stsb`, `mrpc`, `sst2`, `qnli`, `qqp`, `mnli`, `squad`, `squad_v2` | 5e-5, 3e-5, 2e-5 | 16, 32 | 2, 3, 4 |
| GPT-2 | `penn_treebank` | 1e-4 | 32 | 10 |
| GPT-2 | `wikitext-2-v1` | 5e-4 | 32 | 8 |
| GPT-2 | `wikitext-103-v1` | 1e-4 | 32 | 10 |

## Tái thực nghiệm Table 2 trên một số tác vụ

Chúng tôi tái thực hiện bốn điều kiện BERT-base trong Table 2 trên MRPC, STS-B, SST-2, SQuAD 1.1 và QNLI, với tổng cộng 20 kết quả. Mỗi ô ghi điểm trong bài báo → điểm chạy lại, kèm chênh lệch theo điểm phần trăm (chạy lại trừ bài báo). [Table 2 của bài báo](https://aclanthology.org/2024.lrec-main.1162.pdf#page=8)

| Tác vụ (thước đo) | BERTBASE | Random | +MC | +CS |
|:--|--:|--:|--:|--:|
| MRPC (accuracy) | 86.27 → 81.86 (−4.41) | 77.45 → 76.96 (−0.49) | 81.86 → 81.13 (−0.73) | 86.03 → 82.60 (−3.43) |
| STS-B (Spearman) | 88.59 → 87.49 (−1.10) | 85.82 → 82.51 (−3.31) | 86.70 → 84.67 (−2.03) | 88.12 → 86.29 (−1.83) |
| SST-2 (accuracy) | 93.12 → 91.97 (−1.15) | 90.83 → 91.40 (+0.57) | 91.74 → 92.43 (+0.69) | 92.32 → 91.40 (−0.92) |
| SQuAD 1.1 (F1) | 88.48 → 88.03 (−0.45) | 84.79 → 84.09 (−0.70) | 85.37 → 84.86 (−0.51) | 86.70 → 85.55 (−1.15) |
| QNLI (accuracy) | 91.41 → 91.32 (−0.09) | 88.49 → 86.78 (−1.71) | 89.75 → 88.76 (−0.99) | 89.22 → 88.63 (−0.59) |

Mức cải thiện so với cắt tỉa ngẫu nhiên:

| Tác vụ | Bài báo +MC | Tái thực nghiệm +MC | Bài báo +CS | Tái thực nghiệm +CS |
|:--|--:|--:|--:|--:|
| MRPC | +4.41 | +4.17 | +8.58 | +5.64 |
| STS-B | +0.88 | +2.16 | +2.30 | +3.79 |
| SST-2 | +0.91 | +1.03 | +1.49 | 0.00 |
| SQuAD 1.1 | +0.58 | +0.77 | +1.91 | +1.46 |
| QNLI | +1.26 | +1.98 | +0.73 | +1.85 |

Trên QNLI, ba cấu hình cắt tỉa đều giữ 59.57% FLOPs. Thứ tự BERTBASE > +MC > +CS > Random khớp với bài báo. `+MC` và `+CS` chỉ chênh 0.13 điểm phần trăm ở seed 3407. Mức cải thiện +0.73 của `+CS` trong bài báo được tính từ hai điểm công bố (89.22 − 88.49). Bảng gốc in +0.77.

**Thiết lập lần chạy:** `bert-base-uncased`, seed 3407, 3.000 mẫu hiệu chuẩn, 200 token lấy mẫu, λ=5, δ=0.001 và mục tiêu 60% FLOPs tương đối. Fine-tuning dùng learning rate 2e-5 trong 3 epoch; batch size 32 cho GLUE và 12 cho SQuAD; độ dài chuỗi tối đa lần lượt là 128 và 384. Lần chạy dùng Tesla T4. BERTBASE là BERT được fine-tune nhưng không cắt tỉa trong lần chạy của chúng tôi; điểm bài báo được đưa vào để đối chiếu. Bài báo không nêu rõ cấu hình fine-tuning riêng cho từng tác vụ ở Table 2, vì vậy đây là tái thực nghiệm một seed với cấu hình cố định. Điều kiện SQuAD +CS đã chạy riêng và hoàn tất đủ 3 epoch; Exact Match đạt 77.04 (không có trong Table 2).

Với SQuAD, hãy dùng các trường `raw_score` hoặc `eval_metrics_raw` trong báo cáo. Trường `metrics_percent` hiện bị nhân thêm 100 lần.

Lần chạy QNLI v8 hoàn tất cả bốn cấu hình, mỗi cấu hình 3 epoch, trong 3 giờ 28 phút 52 giây trên Tesla T4. Bộ nhớ GPU đạt đỉnh 4.605 MiB.

Báo cáo gốc gồm [lần chạy ban đầu trên bốn tác vụ](experiments/table2_selected_tasks/initial_run_report.json), [lần chạy bổ sung SQuAD +CS](experiments/table2_selected_tasks/squad_cs_run_report.json) và [lần chạy QNLI v8](experiments/table2_selected_tasks/qnli_run_report.json).

## Chạy pruning và fine-tuning thủ công

### Bước 1: Tạo mask

Dùng `mask.py` để tạo `head_mask.pt` và `neuron_mask.pt` cho mô hình đích. Mask được lưu tại `/PATH/TO/OUTPUT/$TASK`.

```bash
cd /PATH/TO/P-pruning
MODEL=bert-base-uncased
TASK=stsb
SAVE_DIR=/PATH/TO/OUTPUT/$TASK
python ./mask.py \
  --model "$MODEL" \
  --save_dir "$SAVE_DIR" \
  --task "$TASK" \
  --max_instances 3000 \
  --pruning_ratio 0.3 \
  --sampling_tokens 200 \
  --lambda_val 5
```

### Bước 2: Fine-tune mô hình đã cắt

```bash
python ./finetune.py \
  --model_name_or_path "$MODEL" \
  --task_name "$TASK" \
  --mask_dir "$SAVE_DIR" \
  --output_dir /PATH/TO/OUTPUT/finetuned \
  --seed 3407 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0
```

## Trích dẫn

```bibtex
@inproceedings{wang2024pruning,
  title={Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models},
  author={Wang, Pingjie and Liu, Hongcheng and Wang, Yanfeng and Wang, Yu},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={13279--13289},
  year={2024}
}
```
