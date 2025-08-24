import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import numpy as np
import os

def print_mean_speedup_table(speedup_df):
    # Compute the mean over labels (matrices) for each method, threads, tag
    group_cols = ['method', 'threads', 'tag'] if 'tag' in speedup_df.columns else ['method', 'threads']
    mean_df = speedup_df.groupby(group_cols)['speedup'].mean().reset_index()
    methods = mean_df['method'].unique()
    threads = sorted(mean_df['threads'].unique())
    tags = mean_df['tag'].unique() if 'tag' in mean_df.columns else ['']
    for tag in tags:
        print(f"\nAVERAGE SPEEDUP PER ALGORITHM (tag={tag}):")
        header = 'Method'.ljust(20) + ''.join([f"{t:>10}t" for t in threads])
        print(header)
        for method in methods:
            row = mean_df[(mean_df['method']==method) & ((mean_df['tag']==tag) if 'tag' in mean_df.columns else True)]
            vals = [f"{row[row['threads']==t]['speedup'].values[0]:10.2f}" if t in row['threads'].values else ' '*10 for t in threads]
            print(method.ljust(20) + ''.join(vals))

def create_time_table(df, benchmark_type, plot_dir):
    """Create a table with execution times for each combination of label, method, tag, and threads"""
    labels = df['label'].unique()
    methods = df['method'].unique()
    tags = df['tag'].unique() if 'tag' in df.columns else ['']
    threads = sorted(df['threads'].unique())
    for label in labels:
        for tag in tags:
            label_tag_data = df[(df['label'] == label) & (df['tag'] == tag)] if 'tag' in df.columns else df[df['label'] == label]
            table_data = []
            row_labels = []
            for method in methods:
                method_data = label_tag_data[label_tag_data['method'] == method]
                if len(method_data) > 0:
                    row = []
                    time_vals = []
                    for thread_count in threads:
                        time_val = method_data[method_data['threads'] == thread_count]['time_ms'] if 'time_ms' in method_data.columns else None
                        # Prefer rel_err if available, otherwise err
                        if 'rel_err' in method_data.columns:
                            err_val = method_data[method_data['threads'] == thread_count]['rel_err']
                        elif 'err' in method_data.columns:
                            err_val = method_data[method_data['threads'] == thread_count]['err']
                        else:
                            err_val = None
                        if time_val is not None and len(time_val) > 0 and err_val is not None and len(err_val) > 0:
                            row.append(f"{time_val.iloc[0]:.0f} ms\n(err={err_val.iloc[0]:.2e})")
                            time_vals.append(time_val.iloc[0])
                        elif time_val is not None and len(time_val) > 0:
                            row.append(f"{time_val.iloc[0]:.0f}")
                            time_vals.append(time_val.iloc[0])
                        else:
                            row.append("-")
                    # Compute the mean of valid times
                    if len(time_vals) > 0:
                        mean_time = np.mean(time_vals)
                        row.append(f"{mean_time:.0f} ms")
                    else:
                        row.append("-")
                    table_data.append(row)
                    row_labels.append(method)
            if table_data:
                fig, ax = plt.subplots(figsize=(13, 6))
                ax.axis('tight')
                ax.axis('off')
                col_labels = [f"{t} threads" for t in threads] + ["Mean"]
                table = ax.table(cellText=table_data,
                               rowLabels=row_labels,
                               colLabels=col_labels,
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(6)
                table.scale(1.2, 1.5)
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(color='white')
                    elif j == -1:
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#f1f1f2')
                plt.title(f'Time Table - {label} ({benchmark_type}, {tag})', pad=20, fontsize=14, fontweight='bold')
                safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
                safe_tag = str(tag).replace(' ', '_')
                # First level: openmp or blas (based on tag)
                first_level = str(tag).lower() if str(tag).lower() in ['openmp', 'blas'] else 'other'
                subdir = 'precision' if benchmark_type == 'fixed_precision' else 'rank'
                out_dir = os.path.join(plot_dir, first_level, subdir, 'time')
                os.makedirs(out_dir, exist_ok=True)
                filename = f"time_table_{benchmark_type}_{safe_label}_{safe_tag}.png"
                plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Saved time table: {os.path.join(first_level, subdir, 'time', filename)}")
def create_speedup_table(speedup_df, benchmark_type, plot_dir):
    """Create a table with speedups for each combination of label and method"""
    
    # Get all unique values
    labels = speedup_df['label'].unique()
    methods = speedup_df['method'].unique()
    tags = speedup_df['tag'].unique() if 'tag' in speedup_df.columns else ['']
    threads = sorted(speedup_df['threads'].unique())
    
    # Create a figure for each label
    for label in labels:
        for tag in tags:
            label_tag_data = speedup_df[(speedup_df['label'] == label) & (speedup_df['tag'] == tag)] if 'tag' in speedup_df.columns else speedup_df[speedup_df['label'] == label]
            # Prepare data for the table
            table_data = []
            row_labels = []
            for method in methods:
                method_data = label_tag_data[label_tag_data['method'] == method]
                if len(method_data) > 0:
                    row = []
                    for thread_count in threads:
                        speedup_val = method_data[method_data['threads'] == thread_count]['speedup']
                        if len(speedup_val) > 0:
                            row.append(f"{speedup_val.iloc[0]:.2f}")
                        else:
                            row.append("-")
                    table_data.append(row)
                    row_labels.append(method)
            if table_data:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.axis('tight')
                ax.axis('off')
                col_labels = [f"{t} threads" for t in threads]
                table = ax.table(cellText=table_data,
                               rowLabels=row_labels,
                               colLabels=col_labels,
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(color='white')
                    elif j == -1:
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#f1f1f2')
                    else:
                        try:
                            # extract only the speedup (before the parenthesis)
                            text = cell.get_text().get_text()
                            speedup_val = float(text.split('\n')[0])
                            if speedup_val >= 1.8:
                                cell.set_facecolor('#d4edda')
                            elif speedup_val >= 1.2:
                                cell.set_facecolor('#fff3cd')
                            elif speedup_val < 1.0:
                                cell.set_facecolor('#f8d7da')
                        except Exception:
                            pass
                plt.title(f'Speedup Table - {label} ({benchmark_type}, {tag})', pad=20, fontsize=14, fontweight='bold')
                safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
                safe_tag = str(tag).replace(' ', '_')
                # First level: openmp or blas (based on tag)
                first_level = str(tag).lower() if str(tag).lower() in ['openmp', 'blas'] else 'other'
                subdir = 'precision' if benchmark_type == 'fixed_precision' else 'rank'
                out_dir = os.path.join(plot_dir, first_level, subdir, 'speedup')
                os.makedirs(out_dir, exist_ok=True)
                filename = f"speedup_table_{benchmark_type}_{safe_label}_{safe_tag}.png"
                plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Saved speedup table: {os.path.join(first_level, subdir, 'speedup', filename)}")

# List of CSV files to process
csv_files = [
    ('build/res_benchmark_fixed_rank_A.csv', 'fixed_rank'),
    ('build/res_benchmark_fixed_precision_A.csv', 'fixed_precision')
]

# Create a folder for plots if it does not exist
plot_dir = 'benchmark_plots'
os.makedirs(plot_dir, exist_ok=True)

# Process each CSV file
for csv_path, benchmark_type in csv_files:
    print(f"Processing {csv_path}...")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        continue

    labels = df['label'].unique()
    tags = df['tag'].unique() if 'tag' in df.columns else ['']

    for label in labels:
        for tag in tags:
            plt.figure(figsize=(8,6))
            subset = df[(df['label'] == label) & (df['tag'] == tag)] if 'tag' in df.columns else df[df['label'] == label]
            for method in subset['method'].unique():
                method_data = subset[subset['method'] == method].sort_values('threads')
                plt.plot(method_data['threads'], method_data['time_ms'], marker='o', label=f"{method} [{tag}]")
            plt.xlabel('Number of threads')
            plt.ylabel('Time (ms)')
            plt.title(f"{label} - {benchmark_type} ({tag})")
            ax = plt.gca()
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_locator(LogLocator(base=2))
            ax.xaxis.set_major_formatter(LogFormatter(base=2))
            plt.legend(title='Method [tag]')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
            safe_tag = str(tag).replace(' ', '_')
            # First level: openmp or blas (based on tag)
            first_level = str(tag).lower() if str(tag).lower() in ['openmp', 'blas'] else 'other'
            subdir = 'precision' if benchmark_type == 'fixed_precision' else 'rank'
            out_dir = os.path.join(plot_dir, first_level, subdir, 'plot')
            os.makedirs(out_dir, exist_ok=True)
            filename = f"plot_{benchmark_type}_{safe_label}_{safe_tag}.png"
            plt.savefig(os.path.join(out_dir, filename))
            plt.close()
            print(f"  Saved: {os.path.join(first_level, subdir, 'plot', filename)}")

    # Create time table for all
    create_time_table(df, benchmark_type, plot_dir)


    # Compute and print average speedup for all present tags, if possible
    speedup_data = []
    for label in labels:
        for tag in tags:
            subset = df[(df['label'] == label) & (df['tag'] == tag)] if 'tag' in df.columns else df[df['label'] == label]
            for method in subset['method'].unique():
                method_data = subset[subset['method'] == method].sort_values('threads')
                if len(method_data) > 0:
                    baseline_time = method_data[method_data['threads'] == 1]['time_ms']
                    if len(baseline_time) > 0:
                        baseline_time = baseline_time.iloc[0]
                        for _, row in method_data.iterrows():
                            speedup = baseline_time / row['time_ms']
                            speedup_data.append({
                                'label': label,
                                'method': method,
                                'threads': row['threads'],
                                'tag': tag,
                                'speedup': speedup,
                                'time_ms': row['time_ms']
                            })
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        create_speedup_table(speedup_df, benchmark_type, plot_dir)
        print_mean_speedup_table(speedup_df)

# === AVERAGE SPEEDUP TABLE PER ALGORITHM OVER 4 MATRICES ===
def print_mean_speedup_table(speedup_df):
    # Compute the mean over labels (matrices) for each method, threads, tag
    group_cols = ['method', 'threads', 'tag'] if 'tag' in speedup_df.columns else ['method', 'threads']
    mean_df = speedup_df.groupby(group_cols)['speedup'].mean().reset_index()
    methods = mean_df['method'].unique()
    threads = sorted(mean_df['threads'].unique())
    tags = mean_df['tag'].unique() if 'tag' in mean_df.columns else ['']
    for tag in tags:
        print(f"\nAVERAGE SPEEDUP PER ALGORITHM (tag={tag}):")
        header = 'Method'.ljust(20) + ''.join([f"{t:>10}t" for t in threads])
        print(header)
        for method in methods:
            row = mean_df[(mean_df['method']==method) & ((mean_df['tag']==tag) if 'tag' in mean_df.columns else True)]
            vals = [f"{row[row['threads']==t]['speedup'].values[0]:10.2f}" if t in row['threads'].values else ' '*10 for t in threads]
            print(method.ljust(20) + ''.join(vals))

print(f"All plots have been saved in the folder '{plot_dir}'")
