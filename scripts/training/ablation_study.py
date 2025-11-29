"""
Estudio de Ablación para PPO-Clip en CarRacing-v3

Este script ejecuta múltiples experimentos para validar que cada componente
de PPO es necesario para el rendimiento del agente.

Experimentos:
1. Baseline: Configuración completa (control)
2. No-Clip: clip_coef = 10.0 (efectivamente sin clipping)
3. No-Entropy: ent_coef = 0.0 (sin exploración)
4. No-GAE: gae_lambda = 1.0 (Monte Carlo puro, sin smoothing)
5. No-Stack: num_stack = 1 (sin memoria temporal)
6. No-Reward-Shaping: reward_shaping = False (rewards crudos)

Uso:
    python scripts/training/ablation_study.py --experiment baseline
    python scripts/training/ablation_study.py --experiment no_clip
    python scripts/training/ablation_study.py --experiment no_entropy
    python scripts/training/ablation_study.py --experiment no_gae
    python scripts/training/ablation_study.py --experiment no_stack
    python scripts/training/ablation_study.py --experiment no_reward_shaping
    python scripts/training/ablation_study.py --experiment all  # Corre todos
"""

import argparse
import sys
import warnings
from pathlib import Path

# Suprimir warnings de pygame/pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ppo_clip import PPOConfig, PPOTrainer
from utils import set_seed
from utils.device import resolve_device


def get_baseline_config() -> PPOConfig:
    """Configuración baseline (ganadora) para el estudio de ablación."""
    return PPOConfig(
        # Training
        total_timesteps=4000000,  
        seed=42,
        
        # Environment
        num_envs=16,
        num_stack=2,
        frame_skip=2,
        max_offroad_seconds=10.0,
        offroad_penalty=None,
        continuous=False,  
        reward_shaping=True,
        
        # PPO Hyperparameters
        num_steps=128,
        num_minibatches=4,
        update_epochs=10,
        learning_rate=0.0001,
        use_lr_scheduler=False,
        lr_end=0.000001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        
        # Device
        device=resolve_device("auto"), 
        torch_deterministic=False,
        
        # Evaluation
        eval_episodes=10,
        eval_interval=50,
        track_eval=True,
        
        # Checkpointing
        save_interval=100,  
        
        # Video Recording
        video_interval_minutes=None,  
        
        # Logging
        log_root=Path("results/tensorboard_logs/ablation/baseline"),
        checkpoint_root=Path("results/models/ablation/baseline"),
        video_root=Path("results/videos/ablation/baseline"),
    )


def get_ablation_config(experiment: str) -> PPOConfig:
    """
    Retorna la configuración para un experimento de ablación específico.
    
    Args:
        experiment: Nombre del experimento ('baseline', 'no_clip', etc.)
    
    Returns:
        PPOConfig con la modificación correspondiente
    """
    config = get_baseline_config()
    
    if experiment == "baseline":
        # No cambios, esta es la configuración de control
        pass
    
    elif experiment == "no_clip":
        # Ablación del clipping: clip_coef muy alto = sin clipping efectivo
        config.clip_coef = 10.0
        config.log_root = Path("results/tensorboard_logs/ablation/no_clip")
        config.checkpoint_root = Path("results/models/ablation/no_clip")
        config.video_root = Path("results/videos/ablation/no_clip")
        print("ABLACIÓN: Clipping desactivado (clip_coef=10.0)")
        print("   Hipótesis: Inestabilidad severa con picos y caídas abruptas en reward.")
        print("   Posible colapso catastrófico de la política por actualizaciones demasiado grandes.")
    
    elif experiment == "no_entropy":
        # Ablación de entropía: sin exploración
        config.ent_coef = 0.0
        config.log_root = Path("results/tensorboard_logs/ablation/no_entropy")
        config.checkpoint_root = Path("results/models/ablation/no_entropy")
        config.video_root = Path("results/videos/ablation/no_entropy")
        print("ABLACIÓN: Entropía desactivada (ent_coef=0.0)")
        print("   Hipótesis: Convergencia prematura a política subóptima y determinista.")
        print("   La entropía de la política caerá rápidamente a valores cercanos a cero.")
    
    elif experiment == "no_gae":
        # Ablación de GAE: Monte Carlo puro (lambda=1.0)
        config.gae_lambda = 1.0
        config.log_root = Path("results/tensorboard_logs/ablation/no_gae")
        config.checkpoint_root = Path("results/models/ablation/no_gae")
        config.video_root = Path("results/videos/ablation/no_gae")
        print("ABLACIÓN: GAE reducido (gae_lambda=1.0, equivalente a Monte Carlo)")
        print("   Hipótesis: Mayor varianza en el cálculo de advantages.")
        print("   Aprendizaje más lento y curvas de reward/loss más ruidosas.")
    
    elif experiment == "no_stack":
        # Ablación de frame stacking: sin memoria temporal
        config.num_stack = 1
        config.log_root = Path("results/tensorboard_logs/ablation/no_stack")
        config.checkpoint_root = Path("results/models/ablation/no_stack")
        config.video_root = Path("results/videos/ablation/no_stack")
        print("ABLACIÓN: Frame stacking desactivado (num_stack=1)")
        print("   Hipótesis: Performance severamente degradada desde el inicio.")
        print("   Sin información de velocidad/aceleración, el agente no puede anticipar curvas.")
    
    elif experiment == "no_reward_shaping":
        # Ablación de reward shaping: rewards crudos del entorno
        config.reward_shaping = False
        config.log_root = Path("results/tensorboard_logs/ablation/no_reward_shaping")
        config.checkpoint_root = Path("results/models/ablation/no_reward_shaping")
        config.video_root = Path("results/videos/ablation/no_reward_shaping")
        print("ABLACIÓN: Reward shaping desactivado")
        print("   Hipótesis: Aumento significativo en value loss por rewards no normalizados.")
        print("   Mayor inestabilidad en las métricas de entrenamiento, posible degradación de performance.")
    
    else:
        raise ValueError(f"Experimento desconocido: {experiment}")
    
    return config


def run_experiment(experiment: str) -> None:
    """Ejecuta un experimento de ablación."""
    print(f"\n{'='*70}")
    print(f"ESTUDIO DE ABLACIÓN: {experiment.upper()}")
    print(f"{'='*70}\n")
    
    config = get_ablation_config(experiment)
    
    # Mostrar configuración
    print("Configuración del experimento:")
    print(f"  - Total timesteps: {config.total_timesteps:,}")
    print(f"  - Seed: {config.seed}")
    print(f"  - clip_coef: {config.clip_coef}")
    print(f"  - ent_coef: {config.ent_coef}")
    print(f"  - gae_lambda: {config.gae_lambda}")
    print(f"  - num_stack: {config.num_stack}")
    print(f"  - reward_shaping: {config.reward_shaping}")
    print(f"  - Logs: {config.log_root}")
    print()
    
    # Fijar seed para reproducibilidad
    set_seed(config.seed)
    
    # Crear y ejecutar trainer
    trainer = PPOTrainer(config)
    trainer.train()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENTO COMPLETADO: {experiment.upper()}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Estudio de ablación para PPO-Clip en CarRacing-v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experimentos disponibles:
  baseline           - Configuración completa (control)
  no_clip            - Sin clipping (clip_coef=10.0)
  no_entropy         - Sin exploración (ent_coef=0.0)
  no_gae             - Sin GAE smoothing (gae_lambda=1.0)
  no_stack           - Sin frame stacking (num_stack=1)
  no_reward_shaping  - Sin reward shaping
  all                - Ejecuta todos los experimentos (incluyendo baseline)
  all_ablations      - Ejecuta solo ablaciones (sin baseline)

Ejemplos:
  python scripts/training/ablation_study.py --experiment baseline
  python scripts/training/ablation_study.py --experiment all
  python scripts/training/ablation_study.py --experiment all_ablations
  python scripts/training/ablation_study.py --experiment all --skip baseline no_stack
        """
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["baseline", "no_clip", "no_entropy", "no_gae", "no_stack", "no_reward_shaping", "all", "all_ablations"],
        help="Experimento de ablación a ejecutar"
    )
    
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        default=[],
        help="Experimentos a saltar (solo con 'all' o 'all_ablations')"
    )
    
    args = parser.parse_args()
    
    if args.experiment == "all":
        experiments = ["baseline", "no_clip", "no_entropy", "no_gae", "no_stack", "no_reward_shaping"]
        experiments = [exp for exp in experiments if exp not in args.skip]
        print(f"\nEjecutando todos los experimentos de ablación ({len(experiments)} total)")
        if args.skip:
            print(f"Saltando: {', '.join(args.skip)}")
        print(f"Tiempo estimado: ~{len(experiments) * 0.5:.1f} horas (asumiendo ~30min por experimento)\n")
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n{'#'*70}")
            print(f"# EXPERIMENTO {i}/{len(experiments)}")
            print(f"{'#'*70}")
            run_experiment(exp)
    
    elif args.experiment == "all_ablations":
        # Solo las ablaciones, sin baseline
        experiments = ["no_clip", "no_entropy", "no_gae", "no_stack", "no_reward_shaping"]
        experiments = [exp for exp in experiments if exp not in args.skip]
        print(f"\nEjecutando solo experimentos de ablación ({len(experiments)} total, sin baseline)")
        if args.skip:
            print(f"Saltando: {', '.join(args.skip)}")
        print(f"Tiempo estimado: ~{len(experiments) * 0.5:.1f} horas (asumiendo ~30min por experimento)\n")
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n{'#'*70}")
            print(f"# EXPERIMENTO {i}/{len(experiments)}")
            print(f"{'#'*70}")
            run_experiment(exp)
    
    else:
        run_experiment(args.experiment)


if __name__ == "__main__":
    main()
