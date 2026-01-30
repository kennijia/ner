# Bad Case Analysis Report

## Summary
The analysis of `bad_case.txt` strongly suggests that the low F1 score for `ACTION` (0.53) is heavily influenced by **Missing Annotations (False Negatives in Ground Truth)** and **Inconsistent Definitions** in the dataset, rather than predominantly model failure.

## Detailed Error Patterns

### 1. The "Right but Wrong" Problem (Missing Annotations)
The model correctly identifies valid actions, but the Golden Label (Ground Truth) marks them as `O`. This means the manual annotation is incomplete.

*   **Case 12**:
    *   **Text**: "全面**排查**各类风险隐患" (Completely inspect various risks)
    *   **Model**: `排查` -> `ACTION` (Correct: this is clearly an action)
    *   **Golden**: `O` (Missed)
*   **Case 46**:
    *   **Text**: "加强对重点区域的**巡视**" (Strengthen inspection of key areas)
    *   **Model**: `巡视` -> `ACTION` (Correct)
    *   **Golden**: `O` (Missed)
*   **Case 7**:
    *   **Text**: "工程**实施**后" (After project implementation)
    *   **Model**: `实施` -> `ACTION`
    *   **Golden**: `O`

**Insight**: Your model has generalized well to understand that "排查", "巡视", "实施" are actions. The low Precision score is partly due to the dataset failing to verify these correct predictions.

### 2. High Ambiguity of "调度" (Dispatch/Scheduling)
The word "调度" appears frequently but is labeled inconsistently, causing model confusion.

*   **Pattern A (Action)**: In Case 25 ("防洪调度"), it is labeled as `ACTION`.
*   **Pattern B (Ignored)**: In Case 10 ("联合调度信息"), it is labeled as `O`.
*   **Pattern C (Object part?)**: In Case 4 ("水库调度矛盾"), the model predicts `OBJ`, Golden predictions are messy.

**Insight**: The model struggles to distinguish when "调度" is a verb (Action) versus when it acts as a noun modifier (part of a concept).

### 3. Noun vs. Activity Confusion
*   **Case 1**: "初期**蓄水**" (Initial water impoundment)
    *   **Model**: Predicts `ACTION`.
    *   **Golden**: `O`.
    *   Context implies a "phase" (noun), but the word itself is a verb.

## Action Plan

1.  **Refine the Golden Set**: The test set contains errors. Evaluating a model against a flawed test set gives misleading metrics. You need to manually review the test set labels for the `ACTION` class.
2.  **Strict Guidelines**: Define if "Noun-Verbs" (like "蓄水" in "period of storage", or "inspection" in "inspection report") should be `ACTION` or `O`.
    *   Current Rule seems to be: Only predicates (main verbs) are actions?
    *   Model behavior: Tags any word with "action" semantics.
3.  **Data Cleaning Script**: Use the previously provided `analyze_actions.py` to find all variations of "调度" and unify their labeling.
