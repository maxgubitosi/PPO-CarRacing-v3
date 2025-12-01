"""
Script para generar gráficos comparativos del estudio de ablación.

Lee los logs de TensorBoard de cada experimento y genera gráficos comparativos
para incluir en el informe.

Uso:
    python scripts/training/plot_ablation_results.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2


def load_tensorboard_json(json_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga datos de un archivo JSON exportado de TensorBoard.
    
    Returns:
        (steps, values): Arrays de numpy con los pasos y valores
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    steps = np.array([item[1] for item in data])
    values = np.array([item[2] for item in data])
    
    return steps, values


def smooth_curve(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Aplica suavizado de media móvil."""
    if len(values) < window:
        return values
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    
    return np.concatenate([values[:window-1], smoothed])


def plot_ablation_comparison(
    experiments: Dict[str, Path],
    metric_name: str,
    output_file: Path,
    ylabel: str,
    title: str,
    window: int = 50
) -> None:
    """
    Genera un gráfico comparativo de múltiples experimentos.
    
    Args:
        experiments: Dict {experiment_name: json_file_path}
        metric_name: Nombre de la métrica (para logging)
        output_file: Ruta donde guardar el gráfico
        ylabel: Etiqueta del eje Y
        title: Título del gráfico
        window: Ventana para suavizado
    """
    plt.figure(figsize=(14, 8))
    
    # Colores y estilos para cada experimento
    colors = {
        'baseline': '#2E7D32',  # Verde oscuro
        'no_clip': '#D32F2F',   # Rojo
        'no_entropy': '#F57C00', # Naranja
        'no_gae': '#7B1FA2',    # Púrpura
        'no_stack': '#0288D1',  # Azul
        'no_reward_shaping': '#C2185B'  # Rosa
    }
    
    labels = {
        'baseline': 'Baseline (PPO completo)',
        'no_clip': 'Sin Clipping (ε=10.0)',
        'no_entropy': 'Sin Entropía (coef=0.0)',
        'no_gae': 'Sin GAE (λ=1.0)',
        'no_stack': 'Sin Frame Stack (n=1)',
        'no_reward_shaping': 'Sin Reward Shaping'
    }
    
    linestyles = {
        'baseline': '-',
        'no_clip': '--',
        'no_entropy': '--',
        'no_gae': '--',
        'no_stack': '--',
        'no_reward_shaping': '--'
    }
    
    for exp_name, json_file in experiments.items():
        if not json_file.exists():
            print(f"Archivo no encontrado: {json_file}")
            continue
        
        try:
            steps, values = load_tensorboard_json(json_file)
            smoothed = smooth_curve(values, window)
            
            plt.plot(
                steps[:len(smoothed)],
                smoothed,
                label=labels.get(exp_name, exp_name),
                color=colors.get(exp_name, None),
                linestyle=linestyles.get(exp_name, '-'),
                linewidth=2.5 if exp_name == 'baseline' else 2,
                alpha=0.9
            )
            
            print(f"{exp_name}: {len(steps)} puntos cargados")
        
        except Exception as e:
            print(f"Error cargando {exp_name}: {e}")
    
    plt.xlabel('Training Steps', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend(loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Crear directorio si no existe
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {output_file}")
    plt.close()


def main():
    # Directorio base de logs de ablación
    base_dir = Path("results/tensorboard_logs/ablation")
    output_dir = Path("results/plot_from_tensorboard/plots/ablation")
    
    print("\n" + "="*70)
    print("GENERACIÓN DE GRÁFICOS DE ABLACIÓN")
    print("="*70 + "\n")
    
    # Definir experimentos (asumiendo que exportaste los JSONs de TensorBoard)
    experiments = {
        'baseline': base_dir / 'baseline' / 'charts_episodic_return.json',
        'no_clip': base_dir / 'no_clip' / 'charts_episodic_return.json',
        'no_entropy': base_dir / 'no_entropy' / 'charts_episodic_return.json',
        'no_gae': base_dir / 'no_gae' / 'charts_episodic_return.json',
        'no_stack': base_dir / 'no_stack' / 'charts_episodic_return.json',
        'no_reward_shaping': base_dir / 'no_reward_shaping' / 'charts_episodic_return.json',
    }
    
    # Gráfico 1: Recompensa Episódica
    print("\nGenerando gráfico de Recompensa Episódica...")
    plot_ablation_comparison(
        experiments,
        'episodic_return',
        output_dir / 'ablation_episodic_return.png',
        'Recompensa Media por Episodio',
        'Estudio de Ablación: Impacto de Componentes PPO en Rendimiento',
        window=50
    )
    
    # Gráfico 2: Loss Total
    print("\nGenerando gráfico de Loss Total...")
    loss_experiments = {
        k: v.parent / 'losses_loss.json' for k, v in experiments.items()
    }
    plot_ablation_comparison(
        loss_experiments,
        'loss',
        output_dir / 'ablation_loss.png',
        'Loss Total',
        'Estudio de Ablación: Loss Durante el Entrenamiento',
        window=60
    )
    
    # Gráfico 3: Entropía
    print("\nGenerando gráfico de Entropía...")
    entropy_experiments = {
        k: v.parent / 'losses_entropy.json' for k, v in experiments.items()
    }
    plot_ablation_comparison(
        entropy_experiments,
        'entropy',
        output_dir / 'ablation_entropy.png',
        'Entropía de la Política',
        'Estudio de Ablación: Exploración Durante el Entrenamiento',
        window=60
    )
    
    # Gráfico 4: Value Loss
    print("\nGenerando gráfico de Value Loss...")
    value_loss_experiments = {
        k: v.parent / 'losses_value_loss.json' for k, v in experiments.items()
    }
    plot_ablation_comparison(
        value_loss_experiments,
        'value_loss',
        output_dir / 'ablation_value_loss.png',
        'Value Loss',
        'Estudio de Ablación: Error del Crítico',
        window=60
    )
    
    print("\n" + "="*70)
    print("GRÁFICOS GENERADOS EXITOSAMENTE")
    print("="*70)
    print(f"\nGráficos guardados en: {output_dir}")
    print("\nPara el informe, usa principalmente:")
    print("  1. ablation_episodic_return.png - Figura principal")
    print("  2. ablation_loss.png - Análisis de convergencia")
    print("  3. ablation_entropy.png - Análisis de exploración")


if __name__ == "__main__":
    main()
