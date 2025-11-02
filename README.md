# PPO-CarRacing-v3

Proyecto Final de I404 - Aprendizaje reforzado: Implementación desde cero de PPO para un agente de CarRacing-v3

## Setup

```bash
cd PPO-CarRacing-v3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Jugar en modo humano

```bash
source .venv/bin/activate
python scripts/random/car_racing_human.py
```

## Entrenamiento

```bash
source .venv/bin/activate
python scripts/training/train_ppo_clip.py --total-timesteps 2000000 --num-envs 4 --num-steps 256
```

TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/ppo_clip
```

Los entrenamientos guardan GIFs periódicos de la política en `videos/ppo_clip/` (visibles también en TensorBoard bajo `policy/sample`).
