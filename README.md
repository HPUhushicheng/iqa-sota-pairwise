# IQA Pairwise SOTA Baseline (Small-Data)

面向你这个场景（约 200 张标注图，A/B 二分类）的稳健方案：

- 冻结视觉特征（CLIP）
- 融合手工 IQA 特征（锐度、噪声、曝光、对比度等）
- Pairwise 特征建模：`f(A)-f(B)` 与 `|f(A)-f(B)|`
- `StratifiedGroupKFold` 5-fold（按 `pair_id` 分组，防止顺序泄漏）
- 分类器融合：`LogisticRegression + HistGBDT + (optional) XGBoost`
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
  --use-lr 1 \
  --use-hgb 1 \
  --use-xgboost 1 \
  --device cuda
```

产物：

- `artifacts/run_clip_pairwise/model_bundle.joblib`
- `artifacts/run_clip_pairwise/cv_report.json`
- `artifacts/run_clip_pairwise/oof_predictions.csv`

## 4. 测试集推理 / 生成提交

假设测试包也有 `data.jsonl` + `images` 或 `image`：

```bash
python scripts/predict.py \
  --model-bundle artifacts/run_clip_pairwise/model_bundle.joblib \
  --dataset test:/path/to/test/data.jsonl:/path/to/test/images \
  --output /path/to/predictions.jsonl \
  --output-style competition \
  --with-thinking 1 \
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

## 6. 可继续提升

1. 开启 `--use-pyiqa 1`，增加 NR-IQA 先验特征。
2. 调整 `clip_model`（例如更大 CLIP backbone）。
3. 对 OOF 概率再做温度缩放或更细阈值搜索。
