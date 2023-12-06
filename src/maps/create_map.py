#!/usr/bin/env python3
import pygame
import sys

# Initialisation de pygame
pygame.init()

# Paramètres de la fenêtre
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Créateur de Carte")

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Liste pour stocker les nœuds et les liens
nodes = []
links = []

def save_to_file(nodes, links, filename="map.txt"):
    with open(filename, 'w') as file:
        # Écrire les nœuds
        for i, node in enumerate(nodes):
            file.write(f"{node.x} {node.y} : {i}\n")
        file.write("===\n")
        # Écrire les liens
        for link in links:
            if link[1] is not None:
                start_node = nodes.index(link[0])
                end_node = nodes.index(link[1])
                file.write(f"{start_node} {end_node}\n")


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                # Ajout d'un lien si l'espace est enfoncé
                for node in nodes:
                    if node.collidepoint(pos):
                        if len(links) > 0 and links[-1][1] is None:
                            links[-1][1] = node
                        else:
                            links.append([node, None])
                        break
            else:
                # Ajout d'un nœud si l'espace n'est pas enfoncé
                nodes.append(pygame.Rect(pos[0], pos[1], 10, 10))
        
    screen.fill(WHITE)
    
    # Dessiner les liens
    for link in links:
        if link[1] is not None:
            pygame.draw.line(screen, BLACK, link[0].center, link[1].center, 2)
    
    # Dessiner les nœuds
    for node in nodes:
        pygame.draw.rect(screen, RED, node)

    pygame.display.flip()


save_to_file(nodes, links)
print("Carte sauvegardée dans 'map.txt'")
pygame.quit()