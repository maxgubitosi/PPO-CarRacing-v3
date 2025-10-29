##########################################################################
# INSTRUCCIONES
##########################################################################
# Jugar CarRacing-v3 como humano 
# Controles: 
#   ←/→ (dirección)
#   ↑ (acelerar) 
#   ↓ (frenar)
#   Q o ESC para salir, R para resetear episodio.
##########################################################################

import numpy as np
import pygame
import gymnasium as gym

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = env.reset(seed=0)

    pygame.init()
    clock = pygame.time.Clock()

    steer = 0.0
    STEER_SPEED = 0.06   # cuánto cambia el volante por frame al mantener flechas
    STEER_DECAY = 0.85   # cuánto “vuelve al centro” cuando no tocás nada
    MAX_STEER = 1.0

    running = true = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False

        if keys[pygame.K_r]:
            obs, info = env.reset()
            steer = 0.0

        # --- Controles humanos ---
        # Dirección con suavizado (rate control)
        if keys[pygame.K_LEFT]:
            steer -= STEER_SPEED
        elif keys[pygame.K_RIGHT]:
            steer += STEER_SPEED
        else:
            steer *= STEER_DECAY  

        steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))

        gas = 1.0 if keys[pygame.K_UP] else 0.0
        brake = 1.0 if keys[pygame.K_DOWN] else 0.0

        action = np.array([steer, gas, brake], dtype=np.float32)  # [steer ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]

        # --- Paso del entorno ---
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            steer = 0.0

        # Limitar FPS para un control estable (el env suele renderizar a ~50-60 fps)
        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
