# Model Evaluation Report

## 1. Performance Overview
*   **Overall F1 Score**: **0.726** (Acceptable baseline)
*   **Best Performing Class**: `VALUE` (F1: **0.891**) - The model is excellent at extracting numerical values and units.
*   **Worst Performing Class**: `ACTION` (F1: **0.529**) - This is the primary bottleneck.

## 2. Problem Diagnosis: The "ACTION" Class
The low F1 score for `ACTION` compared to `ORG` (0.80) and `OBJ` (0.79) strongly indicates **inconsistent data annotation**.

### Inconsistency Examples (based on data observation):
1.  **Granularity Mismatch**:
    *   **Atomic Verbs**: "切换" (Switch), "记录" (Record), "报" (Report).
    *   **Noun Phrases**: "水资源统一调度" (Unified water resources scheduling), "防洪调度" (Flood control scheduling), "水力发电" (Hydroelectric power generation).
    *   **Verb Phrases**: "加固处理" (Reinforcement treatment), "巡视检查" (Inspection check).

2.  **Ambiguity**:
    *   The model interprets "调度" (scheduling) sometimes as a standalone action, and sometimes as part of a longer phrase like "防洪调度". This confusing boundary prevents the model from generalizing well.

## 3. Recommendations

### A. Data Cleaning (High Priority)
You should standardize the annotation of `ACTION` entities in `admin_train.json`.
*   **Option 1 (Recommended)**: **Atomic Approach**. Only label the core verb (e.g., "调度", "检查", "启动"). Move modifiers like "防洪" or "水资源" to other tags (e.g., `OBJ` or `TYPE`) or exclude them.
*   **Option 2**: **Phrase Approach**. Consistently include the noun modifier for specific business terms. Ensure that *every* instance of "调度" preceded by "防洪" is labeled as "防洪调度".

### B. Use the Analysis Script
I have created a script `data/my/analyze_actions.py`. Run it to see all unique `ACTION` labels in your dataset. Look for:
*   Duplicate concepts (e.g., "检查" vs "巡查" vs "巡视检查").
*   Extremely long phrases that should likely be split.

### C. Model Configuration
The logs show a warning: `Model name 'pretrained_bert_models/chinese_roberta_wwm_large_ext/' not found`. Ensure that your `bert_model` path actually contains the `config.json` and `pytorch_model.bin` for the pre-trained weights, otherwise the model might not be initializing from the best state, although the vocabulary seems to load correctly.