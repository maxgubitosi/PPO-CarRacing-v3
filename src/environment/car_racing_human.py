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
import csv
import os
from datetime import datetime

def get_player_name(screen):
    """Pide el nombre del jugador."""
    font_title = pygame.font.Font(None, 48)
    font_input = pygame.font.Font(None, 40)
    
    clock = pygame.time.Clock()
    player_name = ""
    
    while True:
        screen.fill((20, 20, 20))
        
        # Título
        title = font_title.render("Enter Your Name:", True, (255, 255, 255))
        title_rect = title.get_rect(center=(screen.get_width() // 2, 150))
        screen.blit(title, title_rect)
        
        # Input
        input_text = font_input.render(player_name + "_", True, (100, 255, 100))
        input_rect = input_text.get_rect(center=(screen.get_width() // 2, 250))
        screen.blit(input_text, input_rect)
        
        # Instrucción
        instruction = pygame.font.Font(None, 30).render("Press ENTER to continue", True, (150, 150, 150))
        instruction_rect = instruction.get_rect(center=(screen.get_width() // 2, 330))
        screen.blit(instruction, instruction_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and player_name:
                    return player_name
                elif event.key == pygame.K_BACKSPACE:
                    player_name = player_name[:-1]
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.unicode.isprintable() and len(player_name) < 20:
                    player_name += event.unicode
        
        clock.tick(60)


def show_menu(screen, episode_stats=None):
    """Muestra un menú de inicio y retorna True para jugar, False para salir."""
    font_title = pygame.font.Font(None, 74)
    font_option = pygame.font.Font(None, 48)
    font_stats = pygame.font.Font(None, 32)
    
    clock = pygame.time.Clock()
    
    while True:
        screen.fill((20, 20, 20))
        
        # Título
        title = font_title.render("CarRacing-v3", True, (255, 255, 255))
        title_rect = title.get_rect(center=(screen.get_width() // 2, 100))
        screen.blit(title, title_rect)
        
        # Mostrar estadísticas del episodio anterior si existen
        y_offset = 180
        if episode_stats:
            stats_title = font_stats.render("Last Episode Results:", True, (255, 255, 100))
            stats_title_rect = stats_title.get_rect(center=(screen.get_width() // 2, y_offset))
            screen.blit(stats_title, stats_title_rect)
            
            y_offset += 40
            for key, value in episode_stats.items():
                if key not in ['player_name', 'timestamp']:
                    stat_text = font_stats.render(f"{key}: {value}", True, (200, 200, 200))
                    stat_rect = stat_text.get_rect(center=(screen.get_width() // 2, y_offset))
                    screen.blit(stat_text, stat_rect)
                    y_offset += 35
            
            y_offset += 20
        
        # Opciones
        start_text = font_option.render("Press SPACE to Play Again", True, (100, 255, 100))
        start_rect = start_text.get_rect(center=(screen.get_width() // 2, y_offset))
        screen.blit(start_text, start_rect)
        
        quit_text = font_option.render("Press Q or ESC to Quit", True, (255, 100, 100))
        quit_rect = quit_text.get_rect(center=(screen.get_width() // 2, y_offset + 60))
        screen.blit(quit_text, quit_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
        
        clock.tick(60)


def save_stats_to_csv(stats, filename="results/log/human_play_results.csv"):
    """Guarda las estadísticas del episodio en un archivo CSV."""
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Verificar si el archivo ya existe para agregar headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'player_name', 'total_reward', 'steps', 
                      'avg_reward_per_step', 'termination_reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(stats)
    
    print(f"Stats saved to {filename}")


def main():
    pygame.init()
    
    STEER_SPEED = 0.06   # cuánto cambia el volante por frame al mantener flechas
    STEER_DECAY = 0.85   # cuánto "vuelve al centro" cuando no tocás nada
    MAX_STEER = 1.0
    
    # Pedir nombre del jugador al inicio
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("CarRacing-v3 - Player Name")
    player_name = get_player_name(screen)
    
    if not player_name:
        pygame.quit()
        return
    
    # Cerrar ventana del nombre
    pygame.display.quit()
    
    running = True
    env = None
    episode_stats = None
    first_game = True  # Flag para saltar el menú en el primer juego
    
    while running:
        # Mostrar menú solo si no es el primer juego
        if not first_game:
            pygame.init()
            screen = pygame.display.set_mode((600, 500 if episode_stats else 400))
            pygame.display.set_caption("CarRacing-v3 - Menu")
            
            if not show_menu(screen, episode_stats):
                break
            
            # Cerrar ventana del menú
            pygame.display.quit()
        
        first_game = False
        
        # Iniciar el entorno
        if env is None:
            env = gym.make("CarRacing-v3", render_mode="human")
        obs, info = env.reset()
        
        # Reiniciar pygame para el juego
        pygame.init()
        clock = pygame.time.Clock()
        
        steer = 0.0
        playing = True
        
        # Tracking de estadísticas
        total_reward = 0.0
        steps = 0
        positive_rewards = 0  # Contar rewards positivos (tiles visitados)
        
        while playing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                playing = False
                running = False

            if keys[pygame.K_r]:
                obs, info = env.reset()
                steer = 0.0
                total_reward = 0.0
                steps = 0
                positive_rewards = 0

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

            action = np.array([steer, gas, brake], dtype=np.float32)

            # --- Paso del entorno ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Actualizar estadísticas
            total_reward += reward
            steps += 1
            
            # Contar tiles visitados (reward > 0 significa que visitaste un tile)
            if reward > 0:
                positive_rewards += 1

            if terminated or truncated:
                # Determinar la razón de terminación
                if truncated:
                    # Se acabó el tiempo/límite de pasos
                    termination_reason = "timeout"
                elif terminated:
                    # Terminó naturalmente
                    if total_reward < -50:
                        # Crasheó (recibe -100 de penalización)
                        termination_reason = "crashed"
                    else:
                        # Completó el circuito exitosamente
                        termination_reason = "completed"
                else:
                    termination_reason = "unknown"
                
                # Guardar estadísticas
                episode_stats = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'player_name': player_name,
                    'total_reward': round(total_reward, 2),
                    'steps': steps,
                    'avg_reward_per_step': round(total_reward / steps if steps > 0 else 0, 4),
                    'termination_reason': termination_reason
                }
                
                save_stats_to_csv(episode_stats)
                
                # Terminar el loop de juego para volver al menú
                playing = False

            # Limitar FPS para un control estable
            clock.tick(60)
        
        # Cerrar ventana del juego antes de volver al menú
        env.close()
        pygame.quit()
        env = None
    
    pygame.quit()

if __name__ == "__main__":
    main()
