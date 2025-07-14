import gradio as gr
import pandas as pd
import json
from plugins import cleanlab_datalab_plugin
import tempfile

def run_cleanlab_pipeline(file, label_col, issue_types, confidence_threshold):
    if file is None:
        return "Please upload a CSV file.", None, None
    df = pd.read_csv(file.name)
    labels = df[label_col].tolist() if label_col and label_col in df.columns else None
    config = {
        'enable_cleanlab': True,
        'cleanlab_label_column': label_col,
    }
    if issue_types:
        config['cleanlab_issue_types'] = [t.strip() for t in issue_types.split(',')]
    if confidence_threshold:
        try:
            config['cleanlab_confidence_threshold'] = float(confidence_threshold)
        except Exception:
            pass
    result = cleanlab_datalab_plugin.cleanlab_datalab_plugin(df, labels=labels, config=config)
    # Filter out rows with issues
    if result and result.get('issues'):
        issue_rows = set(iss['row'] for iss in result['issues'] if 'row' in iss)
        cleaned_df = df[~df.index.isin(issue_rows)]
    else:
        cleaned_df = df
    # Save outputs to temp files for download
    issues_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8')
    json.dump(result, issues_file, ensure_ascii=False, indent=2)
    issues_file.close()
    cleaned_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8')
    cleaned_df.to_csv(cleaned_file.name, index=False)
    cleaned_file.close()
    # Display issues as DataFrame if present
    issues_df = pd.DataFrame(result['issues']) if result and result.get('issues') else pd.DataFrame()
    return issues_df, cleaned_file.name, issues_file.name

demo = gr.Blocks()
with demo:
    gr.Markdown("# Custom Cleanlab Data Issue Detection WebUI\nUpload a CSV, set options, run Cleanlab, and download results.")
    with gr.Row():
        file_input = gr.File(label="Upload CSV", type="file")
        label_col = gr.Textbox(label="Label Column", value="label")
        issue_types = gr.Textbox(label="Issue Types (comma-separated)", value="label_error,outlier")
        confidence = gr.Textbox(label="Confidence Threshold", value="0.8")
    run_btn = gr.Button("Run Cleanlab Pipeline")
    issues_output = gr.Dataframe(label="Detected Issues", interactive=False)
    cleaned_download = gr.File(label="Download Cleaned CSV")
    issues_download = gr.File(label="Download Issues JSON")
    run_btn.click(run_cleanlab_pipeline, inputs=[file_input, label_col, issue_types, confidence], outputs=[issues_output, cleaned_download, issues_download])

demo.launch(server_name="0.0.0.0", server_port=7862) 