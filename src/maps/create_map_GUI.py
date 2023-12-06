#!/usr/bin/env python3
import pygame
import sys
import os
BUILD_DIR = os.path.join(os.path.dirname(__file__), 'build')

# Initialisation de pygame
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Créateur de Carte")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

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

def remove_node(node_to_remove):
    global links
    links = [link for link in links if link[0] != node_to_remove and link[1] != node_to_remove]
    nodes.remove(node_to_remove)

def remove_link(link_to_remove):
    links.remove(link_to_remove)



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
            elif pygame.key.get_pressed()[pygame.K_d]:
                print("DELETE")
                # Suppression d'un nœud ou d'un lien si la touche D est enfoncée
                clicked_node = next((node for node in nodes if node.collidepoint(pos)), None)
                clicked_link = next((link for link in links if link[1] is not None and pygame.draw.line(screen, BLACK, link[0].center, link[1].center, 2).collidepoint(pos)), None)

                if clicked_node:
                    remove_node(clicked_node)
                elif clicked_link:
                    remove_link(clicked_link)
            else:
                # Ajout d'un nœud simple sinon
                nodes.append(pygame.Rect(pos[0], pos[1], 10, 10))
        
    screen.fill(WHITE)
    
    for link in links:
        if link[1] is not None:
            pygame.draw.line(screen, BLACK, link[0].center, link[1].center, 2)

    for node in nodes:
        pygame.draw.rect(screen, RED, node)

    pygame.display.flip()

pygame.quit()


while True:
    should_save = input("Do you want to save the map? (yes/no) ").lower()
    
    if should_save == 'yes':
        filename = input("Enter the filename to save: ")
        file_path = os.path.join(BUILD_DIR, filename)
        
        if os.path.exists(file_path):
            overwrite = input(f"The file '{filename}' already exists. Do you want to overwrite it? (yes/no) ").lower()
            if overwrite != 'yes':
                continue
        
        save_to_file(nodes, links, file_path)
        print(f"Map saved to '{filename}'")

        generate_paths = input("Do you want to generate the corresponding paths file? (yes/no) ").lower()
        if generate_paths == 'yes':
            path_filename = os.path.join(BUILD_DIR, filename.split('.')[0] + "_paths.txt")
            os.system(f"./src/maps/cpp/dijkstra {file_path} > {path_filename}")
            print(f"Paths file generated: '{path_filename}'")
        break

    elif should_save == 'no':
        break

    else:
        print("Please answer 'yes' or 'no'.")