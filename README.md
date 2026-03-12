# IQA Pairwise SOTA Baseline (Small-Data)

面向你这个场景（约 200 张标注图，A/B 二分类）的稳健方案：

- 冻结视觉特征（CLIP）
- 融合手工 IQA 特征（锐度、噪声、曝光、对比度等）
- Pairwise 特征建模：`f(A)-f(B)` 与 `|f(A)-f(B)|`
- `StratifiedGroupKFold` 5-fold（按 `pair_id` 分组，防止顺序泄漏）
- 分层策略支持 `label+source`（减少跨域 fold 波动）
- 分类器融合：`LogisticRegression + HistGBDT + (optional) XGBoost`
- 自动融合择优：`logreg blender` vs `weighted mean`（OOF 选优）
- 推理阶段 `swap-TTA`：融合 `P(A,B)` 与 `1-P(B,A)`，降低顺序敏感误判
- OOF 自动选阈值，对测试集输出 `A/B`

## 1. 项目结构

- `src/iqa_pairwise/data.py`: 解析多源 JSONL 与图像路径
- `src/iqa_pairwise/features.py`: CLIP + 手工 IQA + 可选 pyiqa 特征
- `src/iqa_pairwise/model.py`: 5-fold 训练、融合、阈值校准
- `src/iqa_pairwise/thinking.py`: 基于特征差异生成 `<thinking>`
- `scripts/train.py`: 训练入口
- `scripts/predict.py`: 推理/提交入口

## 2. 安装

```bash
cd iqa_sota_pairwise
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果你只想先跑轻量版本，可以不装 `xgboost` / `pyiqa`，脚本会自动降级。

## 3. 训练（你的数据）

在仓库根目录执行：

```bash
python scripts/train.py \
  --dataset train1536:/Users/comefly/Desktop/compeitions/others/IQA/1536/data.jsonl:/Users/comefly/Desktop/compeitions/others/IQA/1536/images \
  --dataset validNew:/Users/comefly/Desktop/compeitions/others/IQA/new/data.jsonl:/Users/comefly/Desktop/compeitions/others/IQA/new/image \
  --artifact-dir artifacts/run_clip_pairwise \
  --n-splits 5 \
  --seed 42 \
  --use-handcrafted 1 \
  --use-clip 1 \
  --use-pyiqa 0 \
  --trusted-source train1536 \
  --noisy-source validNew \
  --denoise-action flip \
  --denoise-threshold 0.85 \
  --trusted-weight 1.0 \
  --noisy-weight 0.6 \
  --flipped-weight 0.8 \
  --stratify-by-source 1 \
  --use-lr 1 \
  --use-hgb 1 \
  --use-xgboost 1 \
  --device cuda
```

产物：

- `artifacts/run_clip_pairwise/model_bundle.joblib`
- `artifacts/run_clip_pairwise/cv_report.json`
- `artifacts/run_clip_pairwise/oof_predictions.csv`
- `artifacts/run_clip_pairwise/dropped_samples.csv`（仅当 `--denoise-action drop` 时产生）

## 4. 测试集推理 / 生成提交

假设测试包也有 `data.jsonl` + `images` 或 `image`：

```bash
python scripts/predict.py \
  --model-bundle artifacts/run_clip_pairwise/model_bundle.joblib \
  --dataset test:/path/to/test/data.jsonl:/path/to/test/images \
  --output /path/to/predictions.jsonl \
  --output-style competition \
  --with-thinking 1 \
  --swap-tta 1 \
  --include-prob 0 \
  --device cuda
```

`--output-style` 说明：

- `competition`: 输出 `{"images": [...], "solution": "<thinking>...<answer>A/B</answer>"}`
- `answer`: 输出 `{"images": [...], "answer": "A/B"}`

## 5. 关键注意点

1. 你的 `1536` 与 `new` 存在大量同名文件，但文件内容不同；训练必须把 `source` 纳入样本标识。
2. 不能用普通随机 KFold；必须按 `pair_id` 分组划分，否则验证分数会虚高。
3. 小样本下不建议直接 SFT 微调 VLM，过拟合风险高；冻结特征 + 轻量分类器更稳。
4. 若某个 source 存在标签噪声（例如 `validNew`），建议启用：`--trusted-source train1536 --noisy-source validNew --denoise-action flip`。

## 6. 可继续提升

1. 开启 `--use-pyiqa 1`，增加 NR-IQA 先验特征。
2. 调整 `clip_model`（例如更大 CLIP backbone）。
3. 对 OOF 概率再做温度缩放或更细阈值搜索。

### 6.1 自动扫 noisy/trusted 策略

```bash
bash scripts/sweep_denoise.sh \
  /root/autodl-tmp/train_data/1536/data.jsonl \
  /root/autodl-tmp/train_data/1536/images \
  /root/autodl-tmp/train_data/images/data.jsonl \
  /root/autodl-tmp/train_data/images \
  /root/autodl-tmp/artifacts/sweep_denoise
```

输出会生成 `leaderboard.csv`，自动给出 Top-3 和最佳配置。

### 6.2 直接从 test 文件夹推理（无 data.jsonl）

当测试目录只包含成对文件 `*_c0.*` / `*_c1.*` 时：

```bash
python -u scripts/predict_from_folder.py \
  --model-bundle /root/autodl-tmp/artifacts/run_clip_pairwise/model_bundle.joblib \
  --test-dir /root/autodl-tmp/test \
  --output /root/autodl-tmp/test_answers.jsonl \
  --output-style answer \
  --swap-tta 1 \
  --strict 1
```

输出中固定按 `[c0, c1]` 排列：`answer=A` 表示 `c0` 更好，`answer=B` 表示 `c1` 更好。

### 6.3 多模型平均概率（无 data.jsonl，一次跑完）

```bash
python -u scripts/predict_ensemble_from_folder.py \
  --model-bundle /root/autodl-tmp/artifacts/run_clip_pairwise/model_bundle.joblib \
  --model-bundle /root/autodl-tmp/artifacts/run_clip_pairwise_v2/model_bundle.joblib \
  --model-bundle /root/autodl-tmp/artifacts/sweep_denoise/flip_t065/model_bundle.joblib \
  --test-dir /root/autodl-tmp/test \
  --output /root/autodl-tmp/test_predictions_ensemble.jsonl \
  --output-style answer \
  --swap-tta 1 \
  --strict 1 \
  --device cuda
```

这条命令会在一次运行内完成：配对 -> 每个模型预测 -> 平均概率 -> 输出最终 `answer`。

## 7. 多 Seed 集成（推荐）

### 7.1 训练 3 个 seed

```bash
bash scripts/train_multiseed.sh \
  /root/autodl-tmp/train_data/1536/data.jsonl \
  /root/autodl-tmp/train_data/1536/images \
  /root/autodl-tmp/train_data/images/data.jsonl \
  /root/autodl-tmp/train_data/images \
  /root/autodl-tmp/artifacts/multiseed
```

### 7.2 多模型平均概率推理

```bash
python -u scripts/predict_ensemble.py \
  --model-bundle /root/autodl-tmp/artifacts/multiseed/seed_42/model_bundle.joblib \
  --model-bundle /root/autodl-tmp/artifacts/multiseed/seed_2024/model_bundle.joblib \
  --model-bundle /root/autodl-tmp/artifacts/multiseed/seed_3407/model_bundle.joblib \
  --dataset test:/path/to/test/data.jsonl:/path/to/test/images \
  --output /path/to/predictions.jsonl \
  --output-style competition \
  --with-thinking 1 \
  --swap-tta 1 \
  --include-prob 0
```

## 8. 推送到 GitHub

仓库里已提供脚本（会创建仓库并推送到你的账号）：

```bash
cd iqa_sota_pairwise
./scripts/push_github.sh
```

默认目标仓库：`HPUhushicheng/iqa-sota-pairwise`。
