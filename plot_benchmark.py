def create_time_table(df, benchmark_type, plot_dir):
    """Crea una tabella con i tempi di esecuzione per ogni combinazione di label, metodo, tag e threads"""
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
                    for thread_count in threads:
                        time_val = method_data[method_data['threads'] == thread_count]['time_ms'] if 'time_ms' in method_data.columns else None
                        # Preferisci rel_err se disponibile, altrimenti err
                        if 'rel_err' in method_data.columns:
                            err_val = method_data[method_data['threads'] == thread_count]['rel_err']
                        elif 'err' in method_data.columns:
                            err_val = method_data[method_data['threads'] == thread_count]['err']
                        else:
                            err_val = None
                        if time_val is not None and len(time_val) > 0 and err_val is not None and len(err_val) > 0:
                            row.append(f"{time_val.iloc[0]:.0f} ms\n(err={err_val.iloc[0]:.2e})")
                        elif time_val is not None and len(time_val) > 0:
                            row.append(f"{time_val.iloc[0]:.0f}")
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
                # Primo livello: openmp o blas (in base al tag)
                first_level = str(tag).lower() if str(tag).lower() in ['openmp', 'blas'] else 'other'
                subdir = 'precision' if benchmark_type == 'fixed_precision' else 'rank'
                out_dir = os.path.join(plot_dir, first_level, subdir, 'time')
                os.makedirs(out_dir, exist_ok=True)
                filename = f"time_table_{benchmark_type}_{safe_label}_{safe_tag}.png"
                plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Salvata tabella tempi: {os.path.join(first_level, subdir, 'time', filename)}")
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import numpy as np
import os

def create_speedup_table(speedup_df, benchmark_type, plot_dir):
    """Crea una tabella con gli speedup per ogni combinazione di label e metodo"""
    
    # Ottieni tutti i valori unici
    labels = speedup_df['label'].unique()
    methods = speedup_df['method'].unique()
    tags = speedup_df['tag'].unique() if 'tag' in speedup_df.columns else ['']
    threads = sorted(speedup_df['threads'].unique())
    
    # Crea una figura per ogni label
    for label in labels:
        for tag in tags:
            label_tag_data = speedup_df[(speedup_df['label'] == label) & (speedup_df['tag'] == tag)] if 'tag' in speedup_df.columns else speedup_df[speedup_df['label'] == label]
            # Prepara i dati per la tabella
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
                            # estrai solo lo speedup (prima della parentesi)
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
                # Primo livello: openmp o blas (in base al tag)
                first_level = str(tag).lower() if str(tag).lower() in ['openmp', 'blas'] else 'other'
                subdir = 'precision' if benchmark_type == 'fixed_precision' else 'rank'
                out_dir = os.path.join(plot_dir, first_level, subdir, 'speedup')
                os.makedirs(out_dir, exist_ok=True)
                filename = f"speedup_table_{benchmark_type}_{safe_label}_{safe_tag}.png"
                plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Salvata tabella speedup: {os.path.join(first_level, subdir, 'speedup', filename)}")

# Lista dei file CSV da processare
csv_files = [
    ('build/res_benchmark_fixed_rank_A.csv', 'fixed_rank'),
    ('build/res_benchmark_fixed_precision_A.csv', 'fixed_precision')
]

# Crea una cartella per i plot se non esiste
plot_dir = 'benchmark_plots'
os.makedirs(plot_dir, exist_ok=True)

# Processa ogni file CSV
for csv_path, benchmark_type in csv_files:
    print(f"Processando {csv_path}...")
    
    # Carica i dati
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Errore nel caricamento del file CSV {csv_path}: {e}")
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
            plt.xlabel('Numero di thread')
            plt.ylabel('Tempo (ms)')
            plt.title(f"{label} - {benchmark_type} ({tag})")
            ax = plt.gca()
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_locator(LogLocator(base=2))
            ax.xaxis.set_major_formatter(LogFormatter(base=2))
            plt.legend(title='Metodo [tag]')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
            safe_tag = str(tag).replace(' ', '_')
            # Primo livello: openmp o blas (in base al tag)
            first_level = str(tag).lower() if str(tag).lower() in ['openmp', 'blas'] else 'other'
            subdir = 'precision' if benchmark_type == 'fixed_precision' else 'rank'
            out_dir = os.path.join(plot_dir, first_level, subdir, 'plot')
            os.makedirs(out_dir, exist_ok=True)
            filename = f"plot_{benchmark_type}_{safe_label}_{safe_tag}.png"
            plt.savefig(os.path.join(out_dir, filename))
            plt.close()
            print(f"  Salvato: {os.path.join(first_level, subdir, 'plot', filename)}")

    # Crea tabella tempi per tutti
    create_time_table(df, benchmark_type, plot_dir)

    # Solo per fixed_rank: speedup e tabella speedup
    if benchmark_type == 'fixed_rank':
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

print(f"Tutti i plot sono stati salvati nella cartella '{plot_dir}'")
