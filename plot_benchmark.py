import pandas as pd
import matplotlib.pyplot as plt
import os

# Percorso del file CSV
csv_path = os.path.join('build', 'res_benchmark_fixed_rank.csv')

# Carica i dati
try:
	df = pd.read_csv(csv_path)
except Exception as e:
	print(f"Errore nel caricamento del file CSV: {e}")
	exit(1)

# Trova le tipologie di matrice (label unici)
labels = df['label'].unique()

# Crea una cartella per i plot se non esiste
plot_dir = 'benchmark_plots'
os.makedirs(plot_dir, exist_ok=True)

for label in labels:
	plt.figure(figsize=(8,6))
	subset = df[df['label'] == label]
	# Plot per ogni metodo
	for method in subset['method'].unique():
		method_data = subset[subset['method'] == method]
		plt.plot(method_data['threads'], method_data['time_ms'], marker='o', label=method)
	plt.xlabel('Numero di thread')
	plt.ylabel('Tempo (ms)')
	plt.title(f"{label}")
	plt.legend(title='Metodo')
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	# Salva il plot
	safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('-', '_').replace('.', '_')
	plt.savefig(os.path.join(plot_dir, f"plot_{safe_label}.png"))
	plt.close()

print(f"Plot salvati nella cartella '{plot_dir}'")
