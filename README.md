# PARALLEL ASTEROIDS

## Distributed Systems class project

This project holds all for implementations made for the distributed systems class within the master’s program in systems and computing engineering of the National University of Colombia.

It contains for folders with the different implementations:

• Sequential
• OpenMP
• CUDA
• OpenMPI

The ’Asteroids’ game was created in C++ using the SFML library. The game dynamics consist on a ship that is controlled by the player, which is wandering through space in a field of asteroids, in order to survive the player must use the ship’s projectiles to destroy the asteroids and in this way earn points and survive.

### How to play

With the game open there are two options when starting the game

- Press number key '1': Play with collisions
- Press number key '2': Play without collisions

Then move the ship with the arroy keys and fire with spacebar

### Install Instructions

In order to run the app in any of its implementations ("/Secuencial", "/OpenMP") do the following:

1. Install SFML
2. g++ main.cpp -o asteroids -I"[Disk]:\[SFML_Path]\include" -L"[Disk]:\[SFML_Path]\include\lib" -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio -pg -no-pie -fopenmp
3. ./asteroids
