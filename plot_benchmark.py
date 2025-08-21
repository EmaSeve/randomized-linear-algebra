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
    threads = sorted(speedup_df['threads'].unique())
    
    # Crea una figura per ogni label
    for label in labels:
        label_data = speedup_df[speedup_df['label'] == label]
        
        # Prepara i dati per la tabella
        table_data = []
        row_labels = []
        
        for method in methods:
            method_data = label_data[label_data['method'] == method]
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
            # Crea la figura per la tabella
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            # Crea la tabella
            col_labels = [f"{t} threads" for t in threads]
            table = ax.table(cellText=table_data,
                           rowLabels=row_labels,
                           colLabels=col_labels,
                           cellLoc='center',
                           loc='center')
            
            # Formattazione della tabella
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Colora l'header
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(color='white')
                elif j == -1:  # header column
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f1f1f2')
                else:
                    # Colora le celle in base al valore dello speedup
                    try:
                        speedup_val = float(cell.get_text().get_text())
                        if speedup_val >= 2.0:
                            cell.set_facecolor('#d4edda')  # verde chiaro
                        elif speedup_val >= 1.5:
                            cell.set_facecolor('#fff3cd')  # giallo chiaro
                        elif speedup_val < 1.0:
                            cell.set_facecolor('#f8d7da')  # rosso chiaro
                    except ValueError:
                        pass
            
            plt.title(f'Speedup Table - {label} ({benchmark_type})', pad=20, fontsize=14, fontweight='bold')
            
            # Salva la tabella
            safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
            filename = f"speedup_table_{benchmark_type}_{safe_label}.png"
            plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Salvata tabella speedup: {filename}")

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

    # Trova le tipologie di matrice (label unici)
    labels = df['label'].unique()

    # Crea tabella speedup per questo benchmark
    speedup_data = []
    
    for label in labels:
        plt.figure(figsize=(8,6))
        subset = df[df['label'] == label]
        
        # Plot per ogni metodo
        for method in subset['method'].unique():
            method_data = subset[subset['method'] == method].sort_values('threads')
            plt.plot(method_data['threads'], method_data['time_ms'], marker='o', label=method)
            
            # Calcola speedup per questa combinazione label-method
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
                            'speedup': speedup
                        })
        
        plt.xlabel('Numero di thread')
        plt.ylabel('Tempo (ms)')
        plt.title(f"{label} - {benchmark_type}")

        # Asse X in scala log base 2
        ax = plt.gca()
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.xaxis.set_major_formatter(LogFormatter(base=2))

        plt.legend(title='Metodo')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        # Salva il plot
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
        filename = f"plot_{benchmark_type}_{safe_label}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()
        print(f"  Salvato: {filename}")
    
    # Crea tabella speedup
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        create_speedup_table(speedup_df, benchmark_type, plot_dir)

print(f"Tutti i plot sono stati salvati nella cartella '{plot_dir}'")
