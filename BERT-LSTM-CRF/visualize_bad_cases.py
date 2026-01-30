import ast
import os
import re

# Configuration
INPUT_FILE = '/home/c403/msy/CLUENER2020/BERT-LSTM-CRF/case/bad_case.txt'
OUTPUT_FILE = '/home/c403/msy/CLUENER2020/BERT-LSTM-CRF/case/bad_case_visualization.html'

def parse_list_str(line):
    """Safely parse the string representation of list from the file."""
    try:
        # Find the list part: ['...', '...']
        match = re.search(r'\[.*\]', line)
        if match:
            return ast.literal_eval(match.group(0))
        return None
    except Exception as e:
        print(f"Error parsing line: {line.strip()} - {e}")
        return None

def process_file():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    html_content = []
    # HTML Header and CSS
    html_content.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: "Microsoft YaHei", sans-serif; padding: 20px; background-color: #f5f5f5; }
            .case-card { background: white; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h3 { color: #333; margin-top: 0; }
            .token-container { display: flex; flex-wrap: wrap; gap: 2px; }
            .token-box { 
                border: 1px solid #ddd; 
                padding: 5px; 
                text-align: center; 
                min-width: 40px; 
                border-radius: 4px;
                background-color: #fff;
            }
            .token-char { font-weight: bold; font-size: 1.2em; margin-bottom: 4px; display: block; }
            .token-label { font-size: 0.8em; color: #666; display: block; }
            .diff-error { background-color: #ffebee; border-color: #ef5350; }
            .diff-error .token-label-pred { color: #d32f2f; font-weight: bold; }
            .legend { margin-bottom: 20px; padding: 10px; background: white; border-radius: 5px; }
            .tag { padding: 2px 4px; border-radius: 3px; font-size: 0.8em; }
        </style>
    </head>
    <body>
    <h1>Bad Case Visualization</h1>
    <div class="legend">
        <strong>Legend:</strong> 
        <span style="background:#ffebee; border:1px solid #ef5350; padding:2px 5px;">Red Background</span> = Mismatch between Golden and Prediction.
    </div>
    """)

    current_case = {}
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith('bad case'):
            # Save previous case if exists
            if 'sentence' in current_case:
                html_content.append(render_case(current_case))
            current_case = {'id': line}
        
        elif line.startswith('sentence:'):
            current_case['sentence'] = parse_list_str(line)
        elif line.startswith('golden label:'):
            current_case['golden'] = parse_list_str(line)
        elif line.startswith('model pred:'):
            current_case['pred'] = parse_list_str(line)

    # Add last case
    if 'sentence' in current_case:
        html_content.append(render_case(current_case))

    html_content.append("</body></html>")

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
    
    print(f"Visualization generated successfully: {OUTPUT_FILE}")

def render_case(case):
    sent = case.get('sentence', [])
    gold = case.get('golden', [])
    pred = case.get('pred', [])

    if not sent or not gold or not pred:
        return ""

    if len(sent) != len(gold) or len(sent) != len(pred):
        return f"<div class='case-card'><h3>{case.get('id')} - Length Mismatch Error</h3></div>"

    html = f"<div class='case-card'><h3>{case.get('id')}</h3><div class='token-container'>"
    
    for char, g_label, p_label in zip(sent, gold, pred):
        is_diff = g_label != p_label
        cls = "token-box diff-error" if is_diff else "token-box"
        
        # Simplify labels for display if needed, or keep full
        html += f"""
        <div class="{cls}">
            <span class="token-char">{char}</span>
            <span class="token-label">G: {g_label}</span>
            <span class="token-label token-label-pred">P: {p_label}</span>
        </div>
        """
    
    html += "</div></div>"
    return html

if __name__ == '__main__':
    process_file()
