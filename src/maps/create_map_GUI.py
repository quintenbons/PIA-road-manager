#!/usr/bin/env python3
import argparse
import shutil
import pygame
import os
import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from graphics.constants import *
from graphics.draw import draw_grid, draw_scale

BUILD_DIR = os.path.join(os.path.dirname(__file__), 'build/GUI/')

parser = argparse.ArgumentParser(description="Créateur de Carte avec option d'arrière-plan PNG.")
parser.add_argument("--png", type=str, help="Chemin vers une image PNG à utiliser comme arrière-plan.", default=None)
parser.add_argument("--grid", action='store_true', help="Afficher une grille sur l'écran.")
args = parser.parse_args()

background_image = None
if args.png:
    try:
        background_image = pygame.image.load(args.png)
        background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    except Exception as e:
        print(f"Erreur lors du chargement de l'image de fond : {e}")


pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Créateur de Carte")

LENGTH_SCALE = 100  # Echelle affichée sur l'écran de {LENGTH_SCALE}px
METERS_AMOUNT = 50  # Nombre de mètres représentés par {LENGTH_SCALE}px
scale_ratio = LENGTH_SCALE / METERS_AMOUNT  # Nombre de pixels pour un mètre

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

nodes = []
links = []

def save_to_file(nodes, links, filename):
    with open(filename, 'w') as file:
        for i, node in enumerate(nodes):
            scaled_x = node.x / scale_ratio
            scaled_y = node.y / scale_ratio
            file.write(f"{scaled_x} {scaled_y} : {i}\n")
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
    if background_image:
        screen.blit(background_image, (0, 0))
    if args.grid:
        draw_grid(screen, LENGTH_SCALE, screen.get_size())


    draw_scale(screen, LENGTH_SCALE, METERS_AMOUNT)
    
    for link in links:
        if link[1] is not None:
            pygame.draw.line(screen, BLACK, link[0].center, link[1].center, 2)

    for node in nodes:
        pygame.draw.circle(screen, BLUE, node.center, 5)

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
            if overwrite == 'y':
                shutil.rmtree(os.path.dirname(file_path))
                os.makedirs(os.path.dirname(file_path))
            elif overwrite == 'n':
                continue
            else:
                print("Not a valid answer.")
                exit()

        if os.path.exists(file_path):
            overwrite = input(f"The file '{filename}' already exists. Do you want to overwrite it? (y/n) ").lower()
            if overwrite == 'y':
                os.remove(file_path)
            elif overwrite == 'n':
                continue
            else:
                print("Not a valid answer.")
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
