////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>

#include "Headers/ship.h"
#include "Headers/bulllet.h"
#include "Headers/asteroid.h"

#include "Headers/deviceQuery.hpp"
#include <helper_functions.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include <cstdio>


#define PI 3.1415926536f
const int gameWidth = 1200;
const int gameHeight = 900;

////////////////////////////////////////////////////////////
/// kernel Definitions to consume kernels
////////////////////////////////////////////////////////////

// The CUDA kernel launchers that get called
extern "C" {
    bool cuda_asteroidPos(float* theta, float factor, float* outX, float* outY, int numElems);
    bool cuda_asteroidBounds(float* posX, float* posY, float* w, float* h, float* outX, float* outY, int numElems);
    bool cuda_asteroidCheckBullet(float* posX, float* posY, float* w, float* h, float* bull_l, float* bull_r, float* bull_t, float* bull_b, int* out, int numElems);
}

// The CUDA kernel launchers that get called
void RunAsteroidKernels(std::vector<Asteriod>& asteroidArr, std::vector<Bullet>& bulletArr, float factor) {

    int numElements = asteroidArr.size();

    cudaError_t err = cudaSuccess;
    size_t size = numElements * sizeof(float);
    size_t sizeBool = numElements * sizeof(bool);
    size_t sizeInt = numElements * sizeof(int);

    if (numElements > 0) {
        /*Movement Kernel*/
        {
            // Allocate the host input Angles
            float* h_theta = (float*)malloc(size);
            // Allocate the host output X Coor
            float* h_xCoord = (float*)malloc(size);
            // Allocate the host output Y Coor
            float* h_yCoord = (float*)malloc(size);

            // Verify that allocations succeeded
            if (h_theta == NULL || h_xCoord == NULL || h_yCoord == NULL)
            {
                fprintf(stderr, "Failed to allocate host vectors!\n");
                exit(EXIT_FAILURE);
            }

            /*Initialize theta*/
            for (int i = 0; i < numElements; ++i)
            {
                h_theta[i] = asteroidArr[i].getRotation(); // Get rotation to be operated by kernel;
            }

            // Allocate the device input vector theta
            float* d_theta = NULL;
            err = cudaMalloc((void**)&d_theta, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector theta (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector xCoord 
            float* d_xCoord = NULL;
            err = cudaMalloc((void**)&d_xCoord, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector xCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector yCoord 
            float* d_yCoord = NULL;
            err = cudaMalloc((void**)&d_yCoord, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector yCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Copy thetha host to device
            err = cudaMemcpy(d_theta, h_theta, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector theta from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Execute movement kernel
            cuda_asteroidPos(d_theta, factor, d_xCoord, d_yCoord, numElements);

            //Copy result xCoord from device to host
            err = cudaMemcpy(h_xCoord, d_xCoord, size, cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xCoord from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Copy result yCoord from device to host
            err = cudaMemcpy(h_yCoord, d_yCoord, size, cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xCoord from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //=>Asteroid actions
            for (int i = 0; i < numElements; i++)
            {
                asteroidArr[i].rotate(0.5f);
                asteroidArr[i].move(sf::Vector2f(h_xCoord[i], h_yCoord[i]));
            }

            //Free devide theta memory
            err = cudaFree(d_theta);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector theta (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Free devide xCoord memory
            err = cudaFree(d_xCoord);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector xCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Free devide yCoord memory
            err = cudaFree(d_yCoord);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector yCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Free host memory
            free(h_theta);
            free(h_xCoord);
            free(h_yCoord);
        }

        /*Validate position kernel*/
        {
            // Allocate the host X Pos
            float* h_xPos = (float*)malloc(size);
            // Allocate the host Y Pos
            float* h_yPos = (float*)malloc(size);
            // Allocate the host bounds W
            float* h_wBound = (float*)malloc(size);
            // Allocate the host bounds H
            float* h_hBound = (float*)malloc(size);
            // Allocate the host X Pos
            float* h_xPosCoord = (float*)malloc(size);
            // Allocate the host Y Pos
            float* h_yPosCoord = (float*)malloc(size);


            // Allocate the host bull l
            float* h_bull_l= (float*)malloc(size);
            // Allocate the host bull r
            float* h_bull_r = (float*)malloc(size);
            // Allocate the host bull t
            float* h_bull_t = (float*)malloc(size);
            // Allocate the host bull b
            float* h_bull_b = (float*)malloc(size);
            // Allocate the host bull collision
            int* h_Collision= (int*)malloc(sizeInt);

            


            // Verify that allocations succeeded
            if (h_xPos == NULL || h_yPos == NULL || h_wBound == NULL || h_hBound == NULL || h_bull_l == NULL || h_bull_r == NULL || h_bull_t == NULL || h_bull_b == NULL)
            {
                fprintf(stderr, "Failed to allocate bounds and pos host vectors!\n");
                exit(EXIT_FAILURE);
            }

            /*Initialize pos and bounds*/
            for (int i = 0; i < numElements; ++i)
            {
                // Get position
                h_xPos[i] = asteroidArr[i].getPosition().x; 
                h_yPos[i] = asteroidArr[i].getPosition().y;

                // Get bounds
                h_wBound[i] = asteroidArr[i].getBounds().width; 
                h_hBound[i] = asteroidArr[i].getBounds().height;
            }

            /*Initialize bullet vars*/
            for (int i = 0; i < bulletArr.size(); ++i)
            {
                h_bull_l[i] = bulletArr[i].getLeft();
                h_bull_r[i] = bulletArr[i].getRight();
                h_bull_t[i] = bulletArr[i].getTop();
                h_bull_b[i] = bulletArr[i].getBottom();
            }

            // Allocate the device input vector bounds
            float* d_xPos = NULL;
            err = cudaMalloc((void**)&d_xPos, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector x pos(error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            float* d_yPos = NULL;
            err = cudaMalloc((void**)&d_yPos, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector y pos(error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector bounds
            float* d_wBound = NULL;
            err = cudaMalloc((void**)&d_wBound, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector w bound (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            float* d_hBound = NULL;
            err = cudaMalloc((void**)&d_hBound, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector h bound (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector position X coord
            float* d_xPosCoord = NULL;
            err = cudaMalloc((void**)&d_xPosCoord, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector xCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector position Ycoord
            float* d_yPosCoord = NULL;
            err = cudaMalloc((void**)&d_yPosCoord, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector yCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector bullet L
            float* d_bull_l = NULL;
            err = cudaMalloc((void**)&d_bull_l, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector bull_l (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector bullet R
            float* d_bull_r = NULL;
            err = cudaMalloc((void**)&d_bull_r, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector bull_r (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            
            // Allocate the device input vector bullet T
            float* d_bull_t = NULL;
            err = cudaMalloc((void**)&d_bull_t, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector bull_t (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector bullet B
            float* d_bull_b = NULL;
            err = cudaMalloc((void**)&d_bull_b, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector bull_b (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device input vector bullet collision
            int* d_Collision = NULL;
            err = cudaMalloc((void**)&d_Collision, sizeInt);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector bull collision (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            

            /*Copy posand bounds host to device*/
            err = cudaMemcpy(d_xPos, h_xPos, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xPos from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_yPos, h_yPos, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xPos from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_wBound, h_wBound, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xPos from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_hBound, h_hBound, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xPos from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            /*Copu bullet params*/
            err = cudaMemcpy(d_bull_l, h_bull_l, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector bull_l from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_bull_r, h_bull_r, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector bull_r from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_bull_t, h_bull_t, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector bull_t from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_bull_b, h_bull_b, size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector bull_b from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            ////Execute bounds kernel
            cuda_asteroidBounds(d_xPos, d_yPos, d_wBound, d_hBound, d_xPosCoord, d_yPosCoord, numElements);

            ////Execute collision kernel
            cuda_asteroidCheckBullet(d_xPos, d_yPos, d_wBound, d_hBound, d_bull_l, d_bull_r, d_bull_t, d_bull_b, d_Collision, numElements);

            //Copy result device to host
            err = cudaMemcpy(h_xPosCoord, d_xPosCoord, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector xPosCoord from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = cudaMemcpy(h_yPosCoord, d_yPosCoord, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector yPosCoord from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = cudaMemcpy(h_Collision, d_Collision, sizeInt, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector bullet collision from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //=>Asteroid actions
            for (int i = 0; i < numElements; i++)
            {
                if (h_xPosCoord[i] != 99999 && h_yPosCoord[i] != 99999) {
                    asteroidArr[i].changePosition(sf::Vector2f(h_xPosCoord[i], h_yPosCoord[i]));
                }
            }

            for (int i = 0; i < numElements; i++){
                if (h_Collision[i] != 99999)
                {
                    if (bulletArr.size() > h_Collision[i]) {
                        if (bulletArr[h_Collision[i]].getCollided() == false) {
                            //score += 10;
                            //scoreMessage.setString("SCORE: " + std::to_string(score));
                            bulletArr[h_Collision[i]].collide();            // Prevent bullet to collide again
                            if (asteroidArr[i].getScale() > 1) // In case asteroid is still big enough to be partitionated
                            {
                                Asteriod newAsteroid_1(8, asteroidArr[i].getScale() - 1, asteroidArr[i].getPosition(), gameWidth, gameHeight);
                                asteroidArr.push_back(newAsteroid_1);

                                /*Asteriod newAsteroid_2(8, asteroidArr[i].getScale() - 1, asteroidArr[i].getPosition(), gameWidth, gameHeight);
                                asteroidArr.push_back(newAsteroid_2);*/
                            }
                            //asteroidHitSound.play();
                        }
                    }
                }
            }

            /*for (int i = 0; i < numElements; i++) {
                if (h_Collision[i] != 99999)
                {
                    asteroidArr.erase(asteroidArr.begin() + i);
                }
            }*/

            //Free devide xPos memory
            err = cudaFree(d_xPos);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector xPos (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            
            //Free devide yPos memory
            err = cudaFree(d_yPos);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector xPos (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Free devide w Bound memory
            err = cudaFree(d_wBound);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector wBound (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Free devide h Bound memory
            err = cudaFree(d_hBound);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector hBound (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Free devide d_xPosCoord memory
            err = cudaFree(d_xPosCoord);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector xPosCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //Free devide d_yPosCoord memory
            err = cudaFree(d_yPosCoord);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector yPosCoord (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //Free devide d_bull_l memory
            err = cudaFree(d_bull_l);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector d_bull_l (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //Free devide d_bull_r memory
            err = cudaFree(d_bull_r);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector d_bull_r (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //Free devide d_bull_t memory
            err = cudaFree(d_bull_t);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector d_bull_t (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //Free devide d_bull_b memory
            err = cudaFree(d_bull_b);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector d_bull_b (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            //Free devide d_Collision memory
            err = cudaFree(d_Collision);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device vector d_Collision (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            
            free(h_xPos);
            free(h_yPos);
            free(h_wBound);
            free(h_hBound);
            
            free(h_bull_l);
            free(h_bull_r);
            free(h_bull_t);
            free(h_bull_b);
            free(h_Collision);
        }
    }
}

////////////////////////////////////////////////////////////
/// Entry point of application
///
/// \return Application exit code
///
////////////////////////////////////////////////////////////
int main()
{
    freopen("output.txt", "w+", stdout);
    freopen("error.txt", "w+", stderr);

    std::array<int, 9> cudaInfo = getCudaInfo();
    std::cout << "CUDA Cores: " << cudaInfo[0] << std::endl;
    std::cout << "Block threads: " << cudaInfo[3] << std::endl;
    int blocksPerGrid = (1000 + cudaInfo[3] - 1) / cudaInfo[3];
    std::cout << "Blocks grid: " << blocksPerGrid << std::endl;


    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Define game constants
    const int gameWidth = 1200;
    const int gameHeight = 900;

    // Create the window of the application
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Asteroids - CUDA",
        sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);

    // Load the sounds used in the game
    sf::SoundBuffer asteroidHitBuffer;
    if (!asteroidHitBuffer.loadFromFile("./Resources/hit.wav"))
        return EXIT_FAILURE;
    sf::Sound asteroidHitSound(asteroidHitBuffer);

    // Load the sounds used in the game
    sf::SoundBuffer bulletBuffer;
    if (!bulletBuffer.loadFromFile("./Resources/piu.wav"))
        return EXIT_FAILURE;
    sf::Sound bulletSound(bulletBuffer);
    bulletSound.setVolume(25.f);

    // Load text font
    sf::Font font;
    if (!font.loadFromFile("resources/PressStart2P.ttf"))
        return EXIT_FAILURE;

    // Define game objects
    Ship ship(sf::Vector2f(gameWidth / 2, gameHeight / 2), gameWidth, gameHeight); // Player (ship)

    std::vector<Bullet> bulletArr;     // Bullet Array
    std::vector<Asteriod> asteroidArr; // Asterois Array
    std::vector<bool> asteroidCollision; // Asterois Array

    Asteriod asteroid(8, 4, sf::Vector2f(0, 0), gameWidth, gameHeight);
    asteroidArr.push_back(asteroid);
    asteroidCollision.push_back(false);

    // Control message
    sf::Text controlMessage;
    controlMessage.setFont(font);
    controlMessage.setCharacterSize(40);
    controlMessage.setOrigin(420, 100);
    controlMessage.setPosition(gameWidth / 2, gameHeight / 2);
    controlMessage.setFillColor(sf::Color::White);
    controlMessage.setString("Press '1' to start the\ngame with Collisions\n\n\nPress '2' to start the\ngame without Collisions");

    // Score
    sf::Text scoreMessage;
    scoreMessage.setFont(font);
    scoreMessage.setCharacterSize(40);
    scoreMessage.setPosition(10, 10);
    scoreMessage.setFillColor(sf::Color::White);
    scoreMessage.setString("SCORE: 0");

    // FPS
    sf::Text fpsMessage;
    fpsMessage.setFont(font);
    fpsMessage.setCharacterSize(20);
    fpsMessage.setPosition(500, 10);
    fpsMessage.setFillColor(sf::Color::Blue);
    fpsMessage.setString("FPS: 0");

    // FPS COUNTER
    sf::Text fpsCMessage;
    fpsCMessage.setFont(font);
    fpsCMessage.setCharacterSize(20);
    fpsCMessage.setPosition(1000, 10);
    fpsCMessage.setFillColor(sf::Color::Red);
    fpsCMessage.setString("BAD FPS: 0");

    int score = 0;

    // Define game properties
    sf::Clock gameClock; // Time of game
    sf::Clock clock;     // Time to control smooth movements
    bool isPlaying = false;
    bool hitPause = false;
    bool isCollision = false;

    float fps;
    int fpsCounter;
    sf::Clock fpsClock;
    fpsClock.restart();
    sf::Time previousTime = fpsClock.getElapsedTime();
    sf::Time currentTimeControl;

    const float idleSpeed = 90.f;
    const float bulletSpeed = 25.f;
    float shipDir = 1.f;
    float shipSpeed = 200.f;
    float shipRotation = 3.f;
    bool canFire = true;
    bool isPropulsed = false;
    float lastFire = 0.f;

    float asteroidSpeed = 300.f;
    const float asteroidRotation = 1.5f;
    float lastCreation = 0.f;
    float lastLog = 0.f;

    float maxFps = 0.f;
    float totGoodFps = 0.f;
    int countFps = 0;

    while (window.isOpen())
    {
        // Handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Window closed or escape key pressed: exit
            if ((event.type == sf::Event::Closed) ||
                ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape)))
            {
                float meanFps = totGoodFps / countFps;
                std::cout << "Max FPS:" << maxFps << std::endl;
                std::cout << "Mean FPS:" << meanFps << std::endl;
                window.close();
                break;
            }

            // Enter key pressed: play
            if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Enter || event.key.code == sf::Keyboard::Num1 || event.key.code == sf::Keyboard::Num2))
            {
                if (!isPlaying && (event.key.code == sf::Keyboard::Num1 || event.key.code == sf::Keyboard::Num2))
                {
                    // (re)start the game
                    isPlaying = true;
                    clock.restart();
                    gameClock.restart();

                    fpsCounter = 0;
                    maxFps = 0;
                    // Reset position of ship
                    ship.reset();

                    if (event.key.code == sf::Keyboard::Num1)
                    {
                        isCollision = true;
                    }
                }

                if (hitPause && (event.key.code == sf::Keyboard::Enter))
                {
                    hitPause = false;
                    ship.reset();
                }

                asteroidArr.clear(); // Delete all current asteroids
            }
        }

        //================================================================
        //  FPS CONTROL
        //================================================================
        if (isPlaying)
        {
            currentTimeControl = fpsClock.getElapsedTime();
            float diff = (currentTimeControl.asMilliseconds() - previousTime.asMilliseconds());
            fps = 1000.0f / diff; // the asSeconds returns a float
            previousTime = currentTimeControl;
            fpsMessage.setString("FPS: " + std::to_string(fps));
            if (fps > maxFps) {
                maxFps = fps;
            }
            if (fps < 50)
            {
                fpsMessage.setFillColor(sf::Color::Red);
                fpsCounter++;
                fpsCMessage.setString(std::to_string(fpsCounter));
            }
            else
            {
                totGoodFps += fps;
                countFps++;
                fpsMessage.setFillColor(sf::Color::Blue);
            }
        }


        /*Log control*/
        int currentCheckTime = gameClock.getElapsedTime().asSeconds();
        if (currentCheckTime % 5 == 0 && gameClock.getElapsedTime().asSeconds() > lastLog + 1.f)
        {
            lastLog = gameClock.getElapsedTime().asSeconds();
            std::cout << "asteroids," << asteroidArr.size() << ",lowfps," << fpsCounter << std::endl;
        }


        float deltaTime = clock.restart().asSeconds();
        float factor = asteroidSpeed * deltaTime;

        //================================================================
        //  ASTEROIDS
        //================================================================
        
        //Run asteroid kernels
        RunAsteroidKernels(asteroidArr, bulletArr, factor);

        //=>Asteroid generation
        int currentTime = gameClock.getElapsedTime().asSeconds();
        if (currentTime % 2 == 0 && gameClock.getElapsedTime().asSeconds() > lastCreation + 1.f)
        {
            lastCreation = gameClock.getElapsedTime().asSeconds();
            int xRand = rand() % 1;
            int yRand = rand() % 1;

            int x = 0;
            if (xRand == 0)
            {
                x = gameWidth;
            }

            int y = 0;
            if (yRand == 0)
            {
                y = gameHeight;
            }
            Asteriod newAsteroid(8, 4, sf::Vector2f(x, y), gameWidth, gameHeight);
            asteroidArr.push_back(newAsteroid);
            asteroidCollision.push_back(false);
        }

        if (isPlaying && !hitPause)
        {
            //================================================================
            //  PLAYER (SHIP)
            //================================================================

            //=>Ship's movement
            float shipAngle = (PI / 180) * (ship.getRotation()); // Convert ship's angle to radians
            isPropulsed = false;

            // Idle movement, since it's space ship will always be moving
            float angleX = (idleSpeed * deltaTime * shipDir) * cos(shipAngle);
            float angleY = (idleSpeed * deltaTime * shipDir) * sin(shipAngle);
            ship.move(sf::Vector2f(angleX, angleY));

            // Movement with input keys
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
            {
                float angleX = (shipSpeed * deltaTime) * cos(shipAngle);
                float angleY = (shipSpeed * deltaTime) * sin(shipAngle);
                shipDir = 1; // Set direction of idle movement
                isPropulsed = true;
                ship.move(sf::Vector2f(angleX, angleY));
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
            {
                float angleX = (-shipSpeed * deltaTime) * cos(shipAngle);
                float angleY = (-shipSpeed * deltaTime) * sin(shipAngle);
                shipDir = -1; // Set direction of idle movement
                isPropulsed = true;
                ship.move(sf::Vector2f(angleX, angleY));
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
            {
                ship.rotate(-shipRotation);
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
            {
                ship.rotate(shipRotation);
            }

            //=>Ship's propultion
            if (isPropulsed == true)
            {
                int time = gameClock.getElapsedTime().asSeconds();
                if (time % 2 == 0)
                {
                    ship.switchPropultion();
                }
            }

            //=>Ship's bullets
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space) && gameClock.getElapsedTime().asSeconds() > lastFire + .25f)
            {
                Bullet newBullet(2, ship.getPosition(), PI, ship.getRotation());
                bulletArr.push_back(newBullet);
                bulletSound.play();
                lastFire = gameClock.getElapsedTime().asSeconds(); // Control elapsed time between shots so prevent infinite bullets
            }

            //=>Ship's collisions against asteroids
            for (int i = 0; i < asteroidArr.size(); i++)
            {
                if (asteroidArr[i].shipCollision(ship) == true)
                {
                    if (isCollision == true)
                    {
                        asteroidArr.clear(); // Delete all current asteroids
                        asteroidCollision.clear();
                        controlMessage.setString("Press 'Enter' to\n     revive");
                        hitPause = true;
                    }
                    else
                    {
                        score -= 5;
                        scoreMessage.setString("SCORE: " + std::to_string(score));
                    }
                }
            }

            //================================================================
            //  ASTEROIDS
            //================================================================
            //=>Asteroid actions
            //for (int i = 0; i < asteroidArr.size(); i++)
            //{
            //    //=>Collision
            //    for (int j = 0; j < bulletArr.size(); j++)
            //    {
            //        if (asteroidArr[i].bulletCollision(bulletArr[j]) == true && bulletArr[j].getCollided() == false)
            //        {
            //            score += 10;
            //            scoreMessage.setString("SCORE: " + std::to_string(score));
            //            bulletArr[j].collide();            // Prevent bullet to collide again
            //            if (asteroidArr[i].getScale() > 1) // In case asteroid is still big enough to be partitionated
            //            {
            //                Asteriod newAsteroid_1(8, asteroidArr[i].getScale() - 1, asteroidArr[i].getPosition(), gameWidth, gameHeight);
            //                asteroidArr.push_back(newAsteroid_1);

            //                Asteriod newAsteroid_2(8, asteroidArr[i].getScale() - 1, asteroidArr[i].getPosition(), gameWidth, gameHeight);
            //                asteroidArr.push_back(newAsteroid_2);
            //            }
            //            asteroidArr.erase(asteroidArr.begin() + i);
            //            asteroidHitSound.play();
            //        }
            //    }
            //}

            //================================================================
            //  BULLETS
            //================================================================

            //=>Movement
            for (int i = 0; i < bulletArr.size(); i++)
            {
                bulletArr[i].move(bulletSpeed);
            }

            //=>Check bounds to deletion
            for (int i = 0; i < bulletArr.size(); i++)
            {
                if (bulletArr[i].checkBounds(gameWidth, gameHeight))
                {
                    bulletArr.erase(bulletArr.begin() + i);
                }
            }
        }

        //================================================================
        //  DRAW OBJECTS
        //================================================================

        window.clear(sf::Color(0, 0, 0));

        if (isPlaying && !hitPause)
        {
            ship.draw(window);
            if (isPropulsed == true)
            {
                ship.drawPropultion(window);
            }

            for (int i = 0; i < asteroidArr.size(); i++)
            {
                asteroidArr[i].draw(window);
            }

            for (int i = 0; i < bulletArr.size(); i++)
            {
                bulletArr[i].draw(window);
            }

            window.draw(scoreMessage);
            window.draw(fpsMessage);
            window.draw(fpsCMessage);
        }
        else
        {
            // Draw asteroids for decoration
            for (int i = 0; i < asteroidArr.size(); i++)
            {
                asteroidArr[i].draw(window);
            }

            // Draw the pause message
            window.draw(controlMessage);
        }

        // Display screen
        window.display();
    }

    return EXIT_SUCCESS;
}