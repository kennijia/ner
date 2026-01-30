import json
import os
import sys

def check_data(file_path):
    print(f"正在校验文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    error_count = 0
    warning_count = 0
    total_entities = 0
    
    # 格式统计
    format_stats = {
        "open_interval": 0, # [start, end) 标准 Python 切片
        "closed_interval": 0, # [start, end] 包含末尾
        "mismatch": 0 # 完全对不上
    }

    for idx, line in enumerate(lines):
        line_num = idx + 1
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"[Line {line_num}] JSON 格式错误")
            error_count += 1
            continue

        text = data.get('text', '')
        labels = data.get('label', {})
        
        if not text:
            print(f"[Line {line_num}] 缺少 'text' 字段")
            error_count += 1
            continue

        for label_type, entities in labels.items():
            for entity_text, positions in entities.items():
                for pos in positions:
                    total_entities += 1
                    if not isinstance(pos, list) or len(pos) != 2:
                        print(f"[Line {line_num}] 坐标格式错误: {pos} (实体: {entity_text})")
                        error_count += 1
                        format_stats["mismatch"] += 1
                        continue
                    
                    start, end = pos
                    
                    # 1. 检查越界
                    if start < 0 or end > len(text) + 1: # 宽松一点，允许 end 超出一点点用来检测闭区间
                        print(f"[Line {line_num}] 坐标越界! text长度: {len(text)}, 坐标: {pos}, 实体: {entity_text}")
                        error_count += 1
                        format_stats["mismatch"] += 1
                        continue
                        
                    # 2. 检查内容匹配
                    # CLUENER 标准格式通常是 [start, end) 左闭右开
                    slice_open = text[start:end]
                    # 有些数据集可能是 [start, end] 闭区间
                    slice_closed = text[start:end+1]

                    if slice_open == entity_text:
                        format_stats["open_interval"] += 1
                    elif slice_closed == entity_text:
                        print(f"[Line {line_num}] [警告-格式风险] 坐标疑似由于闭区间 [start, end] 导致不匹配标准格式。实体: '{entity_text}' 坐标: {pos} 切片结果: '{slice_open}' (缺一位)")
                        warning_count += 1
                        format_stats["closed_interval"] += 1
                    else:
                        # 严重错误：完全对不上
                        print(f"[Line {line_num}] [严重错误-内容不符] 实体: '{entity_text}' (类型: {label_type})")
                        print(f"    -> 坐标: {pos}")
                        print(f"    -> 文本切片 [start:end]: '{slice_open}'")
                        print(f"    -> 文本切片 [start:end+1]: '{slice_closed}'")
                        print(f"    -> 原文片段: ...{text[max(0, start-5):min(len(text), end+5)]}...")
                        error_count += 1
                        format_stats["mismatch"] += 1

    print("\n" + "="*30)
    print("校验完成报告")
    print("="*30)
    print(f"处理行数: {len(lines)}")
    print(f"实体总数: {total_entities}")
    print(f"发现错误 (Mismatch/Error): {error_count}")
    print(f"发现警告 (闭区间格式): {warning_count}")
    print("-" * 20)
    print("格式分布:")
    print(f"  标准左闭右开 [start, end): {format_stats['open_interval']}")
    print(f"  闭区间 [start, end]: {format_stats['closed_interval']}")
    print(f"  无法匹配: {format_stats['mismatch']}")
    print("="*30)
    
    if format_stats['closed_interval'] > 0 and format_stats['open_interval'] > 0:
        print("!!! 严重警告: 数据集混合了两种坐标格式，这会导致模型训练极其不稳定！请务必统一！")

if __name__ == "__main__":
    # 默认路径，你可以修改这里或者通过命令行传参
    default_path = '/home/c403/msy/CLUENER2020/BERT-CRF/data/my/admin.json'
    
    target_path = default_path
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        
    check_data(target_path)
