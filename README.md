# PARALLEL ASTEROIDS

## Distributed Systems class project

### Install Instructions

In order to run the app in any of its implementations ("/Secuencial", "/OpenMP") do the following:

1. Install SFML
2. g++ main.cpp -o asteroids -I"[Disk]:\[SFML_Path]\include" -L"[Disk]:\[SFML_Path]\include\lib" -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio -pg -no-pie -fopenmp
3. ./asteroids

### How to play

With the game open there are two options when starting the game

- Press number key '1': Play with collisions
- Press number key '2': Play without collisions

Then move the ship with the arroy keys and fire with spacebar
