"""
Script auxiliar para verificar que los experimentos de ablación se ejecutaron correctamente.

Revisa los directorios de logs y reporta el estado de cada experimento.
"""

from pathlib import Path
from typing import Dict, List


def check_experiment_status() -> Dict[str, dict]:
    """Verifica el estado de cada experimento de ablación."""
    base_dir = Path("results/tensorboard_logs/ablation")
    
    experiments = [
        "baseline",
        "no_clip",
        "no_entropy",
        "no_gae",
        "no_stack",
        "no_reward_shaping"
    ]
    
    status = {}
    
    for exp in experiments:
        exp_dir = base_dir / exp
        status[exp] = {
            "exists": exp_dir.exists(),
            "has_events": False,
            "num_files": 0,
        }
        
        if exp_dir.exists():
            event_files = list(exp_dir.glob("events.out.tfevents.*"))
            status[exp]["has_events"] = len(event_files) > 0
            status[exp]["num_files"] = len(event_files)
    
    return status


def print_status_report(status: Dict[str, dict]) -> None:
    """Imprime un reporte del estado de los experimentos."""
    print("\n" + "="*70)
    print("REPORTE DE ESTADO - ESTUDIO DE ABLACIÓN")
    print("="*70 + "\n")
    
    total = len(status)
    completed = sum(1 for s in status.values() if s["has_events"])
    
    for exp_name, exp_status in status.items():
        status_symbol = "✓" if exp_status["has_events"] else "✗"
        status_text = "COMPLETADO" if exp_status["has_events"] else "PENDIENTE"
        
        print(f"{status_symbol} {exp_name:20s} - {status_text}")
        if exp_status["exists"]:
            print(f"  └─ Archivos de eventos: {exp_status['num_files']}")
    
    print("\n" + "-"*70)
    print(f"Progreso: {completed}/{total} experimentos completados ({completed/total*100:.1f}%)")
    print("-"*70 + "\n")
    
    if completed == total:
        print("✓ ¡Todos los experimentos completados!")
        print("\n📊 Próximos pasos:")
        print("  1. Exportar métricas de TensorBoard como JSON")
        print("  2. Ejecutar: python scripts/training/plot_ablation_results.py")
    else:
        pending = [name for name, s in status.items() if not s["has_events"]]
        print(f"⚠️  Experimentos pendientes: {', '.join(pending)}")
        print("\n📝 Para ejecutar los pendientes:")
        for exp in pending:
            print(f"  python scripts/training/ablation_study.py --experiment {exp}")


def check_json_exports() -> None:
    """Verifica si se exportaron los JSONs de TensorBoard."""
    print("\n" + "="*70)
    print("VERIFICACIÓN DE EXPORTACIÓN JSON")
    print("="*70 + "\n")
    
    base_dir = Path("results/tensorboard_logs/ablation")
    experiments = ["baseline", "no_clip", "no_entropy", "no_gae", "no_stack", "no_reward_shaping"]
    metrics = ["charts_episodic_return", "losses_loss", "losses_entropy", "losses_value_loss"]
    
    all_exported = True
    
    for exp in experiments:
        print(f"\n{exp}:")
        for metric in metrics:
            json_file = base_dir / exp / f"{metric}.json"
            exists = json_file.exists()
            symbol = "✓" if exists else "✗"
            print(f"  {symbol} {metric}.json")
            
            if not exists:
                all_exported = False
    
    print("\n" + "-"*70)
    
    if all_exported:
        print("✓ Todos los JSONs exportados correctamente")
        print("\n📊 Puedes generar los gráficos con:")
        print("  python scripts/training/plot_ablation_results.py")
    else:
        print("⚠️  Algunos JSONs faltan. Instrucciones:")
        print("\n1. Inicia TensorBoard:")
        print("   tensorboard --logdir=results/tensorboard_logs/ablation")
        print("\n2. En la interfaz web, para cada métrica:")
        print("   - Selecciona la métrica")
        print("   - Click en ⋮ (tres puntos)")
        print("   - Download as JSON")
        print("   - Guarda en: results/tensorboard_logs/ablation/<experiment>/<metric>.json")
        print("\nMétricas a exportar:")
        for metric in metrics:
            print(f"  - {metric}")
    
    print("-"*70 + "\n")


def main():
    print("\n🔬 Verificación del Estudio de Ablación")
    
    # Verificar estado de experimentos
    status = check_experiment_status()
    print_status_report(status)
    
    # Verificar exportación de JSONs
    completed = sum(1 for s in status.values() if s["has_events"])
    if completed == len(status):
        check_json_exports()


if __name__ == "__main__":
    main()
