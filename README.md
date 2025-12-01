# PPO-CarRacing-v3

Implementación desde cero de **Proximal Policy Optimization (PPO)** para entrenar un agente en el entorno CarRacing-v3 de Gymnasium.

**Proyecto Final de I404 - Aprendizaje Reforzado**

## Características

- Implementación modular de PPO-Clip siguiendo el paper original
- Soporte para acciones discretas y continuas
- Entrenamiento paralelo con múltiples entornos vectorizados
- Preprocesamiento visual optimizado (grayscale, frame stacking)
- GAE (Generalized Advantage Estimation) para reducir varianza
- Configuración flexible mediante archivos YAML
- Logging extensivo con TensorBoard
- Checkpointing y resumption de entrenamientos

## Requisitos

- Python 3.10+
- CUDA (opcional, para GPU)
- Dependencias en `requirements_venv.txt` o `requirements_conda.txt`

## Instalación

### Con virtualenv

```bash
cd PPO-CarRacing-v3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_venv.txt
```

### Con conda

```bash
conda create -n ppo_carracing python=3.10
conda activate ppo_carracing
pip install -r requirements_conda.txt
```

## Estructura del Proyecto

```
PPO-CarRacing-v3/
├── configs/                         
│   ├── ppo_config.yaml             
│   ├── ppo_config_sota.yaml        
│   └── ppo_config_cont.yaml        
│
├── src/                           # Código fuente
│   ├── ppo_clip/                  # Implementación de PPO
│   │   ├── agent.py               # Agente PPO (policy + value)
│   │   ├── config.py              # Dataclass de configuración
│   │   ├── trainer.py             # Training loop principal
│   │   ├── rollout_buffer.py      # Buffer de experiencias + GAE
│   │   └── networks_*.py          # Arquitecturas de redes neuronales
│   │
│   ├── environment/               # Wrappers del entorno
│   │   └── carracing.py           # Preprocesamiento, frame stacking, etc.
│   │
│   ├── latent/                    # Experimentos con reducción dimensional
│   │   ├── reducers.py            # PCA, VAE
│   │   └── pca_ppo/               # PPO con observaciones PCA
│   │
│   └── utils/                    
│       ├── device.py             
│       └── seed.py               
│
├── scripts/
│   ├── training/                  
│   │   ├── train_with_config.py  
│   │   └── ablation_study.py     
│   │
│   ├── latent_space_experiment/   
│   └── random/                    # Utilidades y demos
│       ├── car_racing_human.py    
│       └── generate_gif_from_model.py
│
├── results/                       
│   ├── models/                   # Checkpoints guardados
│   ├── tensorboard_logs/         
│   ├── videos/                   
│   └── plot_from_tensorboard/    
│
├── train.sh                       # Script de entrenamiento rápido
└── requirements_*.txt             
```

## Uso Rápido

### Jugar Manualmente

```bash
source .venv/bin/activate
python scripts/random/car_racing_human.py
```

Controles: flechas del teclado (izquierda/derecha/arriba/abajo).

### Entrenar un Agente

**Método 1: Con script shell (más rápido)**

```bash
chmod +x train.sh
./train.sh
```

**Método 2: Con configuración YAML**

```bash
python scripts/training/train_with_config.py --config configs/ppo_config.yaml
```

**Método 3: Sobrescribir parámetros desde CLI**

```bash
python scripts/training/train_with_config.py \
    --config configs/ppo_config.yaml \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --seed 42
```

### Visualizar Entrenamiento

```bash
tensorboard --logdir results/tensorboard_logs/ppo_clip
```

Abre el navegador en `http://localhost:6006`

## Configuración

### Archivo YAML

Edita `configs/ppo_config.yaml` para ajustar hiperparámetros:

```yaml
# Entrenamiento
total_timesteps: 12000000
seed: 42

# Entorno
num_envs: 16              # Número de entornos paralelos
num_stack: 2              # Frames apilados (para capturar movimiento)
frame_skip: 2             # Frames salteados entre apilados
discrete: true            # true: Discrete(5), false: Box(3)
reward_shaping: true      # Clip rewards positivos a +1

# Hiperparámetros PPO
num_steps: 128            # Steps por rollout antes de actualizar
num_minibatches: 4        # División del batch para updates
update_epochs: 10         # Epochs sobre el batch completo
learning_rate: 0.0003     # Learning rate inicial
gamma: 0.99               # Discount factor
gae_lambda: 0.95          # GAE λ (bias-variance tradeoff)
clip_coef: 0.2            # Epsilon para PPO-Clip
ent_coef: 0.01            # Bonus de entropía (exploración)
value_coef: 0.5           # Peso de value loss

# Evaluación y checkpointing
eval_episodes: 10         # Episodios por evaluación
eval_interval: 50         # Evaluar cada N updates
save_interval: 50         # Guardar checkpoint cada N updates
```

### Configuraciones Predefinidas

- **`ppo_config.yaml`**: Configuración estándar (discrete, 12M timesteps)
- **`ppo_config_sota.yaml`**: Configuración optimizada para mejor rendimiento
- **`ppo_config_cont.yaml`**: Para acciones continuas

## Outputs del Entrenamiento

Cada run genera:

```
results/
├── models/ppo_clip/<run_name>/
│   ├── ppo_clip_update_50.pt      # Checkpoints periódicos
│   ├── ppo_clip_update_100.pt
│   └── ppo_clip_final.pt          # Checkpoint final
│
├── tensorboard_logs/ppo_clip/<run_name>/
│   └── events.out.tfevents.*      # Logs de TensorBoard
│
└── videos/ppo_clip/<run_name>/
    └── *.gif                       # Videos de episodios
```

## Métricas en TensorBoard

### Durante Entrenamiento

- **`rollout/episode_return`**: Reward total por episodio
- **`rollout/episode_length`**: Duración del episodio en steps
- **`train/policy_loss`**: Pérdida de la política (PPO-Clip objective)
- **`train/value_loss`**: Pérdida del crítico (MSE)
- **`train/entropy`**: Entropía de la política (exploración)
- **`train/approx_kl`**: KL divergence aproximada (cambio de política)
- **`charts/learning_rate`**: Learning rate actual (si se usa scheduler)

### Durante Evaluación

- **`eval/return_mean`**: Reward promedio
- **`eval/return_std`**: Desviación estándar de rewards
- **`eval/episode_length`**: Duración promedio
- **`eval/death_rate`**: Proporción de episodios terminados por salirse de pista

### Otros

- **`actions/distribution`**: Histograma de acciones tomadas (discrete)
- **`policy/sample`**: Videos de episodios (pestaña IMAGES en TensorBoard)

## Reanudar Entrenamiento

### Método 1: Desde YAML

Edita el campo `resume` en tu configuración:

```yaml
resume: "results/models/ppo_clip/ppo_clip_20251130-153045/ppo_clip_update_100.pt"
```

Luego ejecuta normalmente:

```bash
python scripts/training/train_with_config.py --config configs/ppo_config.yaml
```

### Método 2: Desde CLI

```bash
python scripts/training/train_with_config.py \
    --config configs/ppo_config.yaml \
    --resume results/models/ppo_clip/<run_name>/ppo_clip_update_<N>.pt
```

El entrenamiento continuará desde el checkpoint, manteniendo:
- Pesos del modelo
- Estado del optimizer (momentum, etc.)
- Contadores de updates y timesteps

## Experimentos Adicionales

### Ablation Study

Estudia el impacto de cada componente de PPO:

```bash
python scripts/training/ablation_study.py
```

Genera runs sin: clipping, GAE, entropy, reward shaping, frame stacking.

### Latent Space Experiments

Pipeline para entrenar con PCA en lugar de CNN:

```bash
# 1. Recolectar samples
python scripts/latent_space_experiment/1_collect_samples.py

# 2. Entrenar modelos de reducción (PCA, VAE)
python scripts/latent_space_experiment/2_train_latent_models.py

# 3. Analizar espacios latentes
python scripts/latent_space_experiment/3_analyze_latent_spaces.py

# 4. Entrenar agente con PCA
python scripts/latent_space_experiment/4_train_pca_ppo_agent.py

# 5. Generar visualizaciones
python scripts/latent_space_experiment/5_generate_gif_from_model.py
```