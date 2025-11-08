# PPO-CarRacing-v3

Proyecto Final de I404 - Aprendizaje reforzado: Implementación desde cero de PPO para un agente de CarRacing-v3

## Setup

```bash
cd PPO-CarRacing-v3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estructura del Proyecto

```
PPO-CarRacing-v3/
├── results/                          
│   ├── models/            
│   ├── tensorboard_logs/
│   └── videos/             
├── scripts/   
│   ├── latent_space_experiment/            
│   ├── random/
│   └── training/             
├── src/       
│   ├── environment/            
│   ├── latent/
│   ├── ppo_clip/
│   └── utils/             
└── requirements.txt
```

## Jugar en modo humano

```bash
source .venv/bin/activate
python scripts/random/car_racing_human.py
```

## Entrenamiento

**Configuración SOTA recomendada:**

```bash
source .venv/bin/activate
python scripts/training/train_ppo_clip.py \
    --total-timesteps 5000000 \
    --num-envs 8 \
    --num-steps 512 \
    --num-minibatches 4 \
    --update-epochs 10 \
    --learning-rate 1e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.2 \
    --ent-coef 0.01 \
    --value-coef 0.5 \
    --max-grad-norm 0.5 \
    --discrete \
    --eval-interval 50 \
    --save-interval 50
```

**Entrenamiento rápido (para pruebas):**

```bash
python scripts/training/train_ppo_clip.py \
    --total-timesteps 20000 \
    --num-envs 4 \
    --discrete \
    --eval-interval 10
```

## Visualización con TensorBoard

```bash
tensorboard --logdir results/tensorboard_logs/ppo_clip
```

Los entrenamientos guardan:
- **Checkpoints** en `results/models/ppo_clip/<run_name>/`
- **Logs de TensorBoard** en `results/tensorboard_logs/ppo_clip/<run_name>/`
- **Videos/GIFs** en `results/videos/ppo_clip/<run_name>/` (también visibles en TensorBoard bajo `policy/sample`)

## Métricas en TensorBoard

El entrenamiento registra las siguientes métricas:

**Durante el entrenamiento:**
- `rollout/episode_return`
- `rollout/episode_length`
- `losses/*`: Pérdidas del actor, crítico, entropía y total
- `charts/learning_rate`

**Durante la evaluación:**
- `eval/return_mean`, `eval/return_max`, `eval/return_min`
- `eval/death_rate`: % de episodios donde el auto se alejó demasiado de la pista (-100 reward)
- `eval/episode_length`

**Distribución de acciones** 
- `actions/distribution`: Histograma de las 5 acciones tomadas (para el caso discreto)

## Reanudar Entrenamiento

```bash
python scripts/training/train_ppo_clip.py \
    --resume results/models/ppo_clip/<run_name>/ppo_clip_update_<N>.pt \
    --total-timesteps 2000000
```
