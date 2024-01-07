#!/usr/bin/env python3
import pygame
import sys
import os
BUILD_DIR = os.path.join(os.path.dirname(__file__), 'build/GUI/')

# Initialisation de pygame
pygame.init()

WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Créateur de Carte")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

nodes = []
links = []

def save_to_file(nodes, links, filename):
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

        file.write(f"===\nmedium\n10\nuniform\n")

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
                            if links[-1][0] != node:
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
    should_save = input("Do you want to save the map? (y/n) ").lower()
    
    if should_save == 'y':
        filename = input("Enter the filename to save: ")
        file_path = os.path.join(BUILD_DIR+filename, "map.csv")

        # Create the folder for this city if it doesn't exist, and if it does ask the user if he wants to overwrite it
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        else:
            overwrite = input(f"The folder '{file_path}' already exists. Do you want to overwrite it? (y/n) ").lower()
            if overwrite != 'y':
                continue
            else:
                os.remove(file_path)
        
        if os.path.exists(file_path):
            overwrite = input(f"The file '{filename}' already exists. Do you want to overwrite it? (y/n) ").lower()
            if overwrite != 'y':
                continue
            else:
                exit()
        
        save_to_file(nodes, links, file_path)
        print(f"Map saved to '{filename}'")

        generate_paths = input("Do you want to generate the corresponding paths file? (y/n) ").lower()
        if generate_paths == 'y':
            path_filename = os.path.join(BUILD_DIR+filename, "paths.csv")
            os.system(f"./src/maps/cpp/dijkstra {file_path} > {path_filename}")
            print("commande: ", f"./src/maps/cpp/dijkstra {file_path}/map.csv > {path_filename}")
            print(f"Paths file generated: '{path_filename}'")
        break

    elif should_save == 'n':
        break

    else:
        print("Please answer 'y' or 'n'.")
