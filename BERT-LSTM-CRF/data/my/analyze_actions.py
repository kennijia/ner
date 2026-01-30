import json
import os
from collections import Counter

def analyze_actions(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    actions = []
    print(f"Analyzing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                labels = data.get('label', {})
                action_label = labels.get('ACTION', {})
                # ACTION is a dictionary where keys are texts or lists
                # Based on the file provided, format is "ACTION": {"text": [[start, end]]}
                for text in action_label.keys():
                    actions.append(text)
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {i+1}")
    
    counter = Counter(actions)
    
    print(f"\nTotal ACTION entities found: {len(actions)}")
    print(f"Unique ACTION types: {len(counter)}")
    
    print("\n" + "="*40)
    print("Top 20 Most Frequent ACTIONs")
    print("="*40)
    for action, count in counter.most_common(20):
        print(f"{action:<20} : {count}")
        
    print("\n" + "="*40)
    print("Long ACTIONs (Len > 4) - Check for Phrases")
    print("="*40)
    long_actions = sorted([a for a in counter.keys() if len(a) > 4])
    if not long_actions:
        print("No actions longer than 4 characters found.")
    else:
        for action in long_actions:
            print(f"{action} (Count: {counter[action]})")

if __name__ == "__main__":
    # Assuming the script is run from data/my/ or the project root
    target_file = 'admin_train.json' 
    if not os.path.exists(target_file):
        # Try full path if relative fails
        target_file = '/home/c403/msy/CLUENER2020/BERT-LSTM-CRF/data/my/admin_train.json'
    
    analyze_actions(target_file)
