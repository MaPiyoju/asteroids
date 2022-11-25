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

#include <cstdio>

#include <mpi.h>

////////////////////////////////////////////////////////////
/// Entry point of application
///
/// \return Application exit code
///
////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    int myid, numprocs, I, rc, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    

    freopen("output.txt", "w+", stdout);
    freopen("error.txt", "w+", stderr);

    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Define game constants
    const float pi = 3.14159f;
    const int gameWidth = 1200;
    const int gameHeight = 900;


    bool mainView = true;

    // Create the window of the application
    
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Asteroids - MPI",
        sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);


    if (myid != 0) {
        window.close();
        mainView = false;
    }

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
    
    float astAngle[10000];
    float astPosX[10000];
    float astPosY[10000];

    if (myid == 0) {
        Asteriod asteroid(8, 4, sf::Vector2f(0, 0), gameWidth, gameHeight);
        asteroidArr.push_back(asteroid);
    }

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
    int fpsCounter = 0;
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

    
    while (mainView == true || myid != 0)
    {
        unsigned int partition = 0;
        unsigned int astSize = 0;

        if (myid == 0) {       
            astSize = asteroidArr.size();
            partition = astSize / numprocs;
            
            for (int i = 0; i < astSize; i++) {
                astAngle[i] = asteroidArr[i].getRotation();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 1; i < numprocs; i++) {
                MPI_Send(&partition, 1, MPI_INT,i, 0, MPI_COMM_WORLD);
                MPI_Send(&astSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                
                for (int j = i * partition; j <= i * partition + partition; j++) {
                    MPI_Send(&astAngle[j], partition, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Recv(&partition, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&astSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = myid * partition; i <= myid * partition + partition; i++) {
                MPI_Recv(&astAngle[i], partition, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        // Handle events
        sf::Event event;
        if (myid == 0) {
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
                        maxFps = 0;
                        fpsCounter = 0;
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
        }
        //================================================================
        //  FPS CONTROL
        //================================================================
        if (myid == 0) {
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
        }
        //================================================================
        //  ASTEROIDS
        //================================================================
        float deltaTime = clock.restart().asSeconds();
        float factor = asteroidSpeed * deltaTime;
        
        //=>Asteroid actions
        if (myid > 0) {
            for (int i = myid * partition; i <= myid * partition + partition; i++)
            {
                ////=>Movement
                float angleTmp = (pi / 180)* astAngle[i];
                astPosX[i] = std::cos(angleTmp) * factor;
                astPosY[i] = std::sin(angleTmp) * factor;
            }
        }

        if (myid > 0) {
            MPI_Barrier(MPI_COMM_WORLD);

            for (int i = myid * partition; i <= myid * partition + partition; i++) {
                MPI_Send(&astPosX[i], partition, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&astPosY[i], partition, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);

            for (int i = 1; i < numprocs; i++) {
                for (int j = i * partition; j <= i * partition + partition; j++) {
                    MPI_Recv(&astPosX[j], partition, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&astPosY[j], partition, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (myid == 0) {
            //=>Asteroid movement
            for (int i = 0; i <= asteroidArr.size(); i++)
            {
                if (i < asteroidArr.size()) {
                    asteroidArr[i].rotate(0.5f);
                    asteroidArr[i].move(sf::Vector2f(astPosX[i], astPosY[i]));
                }
            }

            //=>Asteroid generation
            int currentTime = gameClock.getElapsedTime().asSeconds();
            if (myid == 0 && currentTime % 2 == 0 && gameClock.getElapsedTime().asSeconds() > lastCreation + 0.5f)
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
            }
        }

        if (isPlaying && !hitPause)
        {
            //================================================================
            //  PLAYER (SHIP)
            //================================================================

            //=>Ship's movement
            float shipAngle = (pi / 180) * (ship.getRotation()); // Convert ship's angle to radians
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
                Bullet newBullet(2, ship.getPosition(), pi, ship.getRotation());
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
            for (int i = 0; i < asteroidArr.size(); i++)
            {
                //=>Collision
                for (int j = 0; j < bulletArr.size(); j++)
                {
                    if (asteroidArr[i].bulletCollision(bulletArr[j]) == true && bulletArr[j].getCollided() == false)
                    {
                        score += 10;
                        scoreMessage.setString("SCORE: " + std::to_string(score));
                        bulletArr[j].collide();            // Prevent bullet to collide again
                        if (asteroidArr[i].getScale() > 1) // In case asteroid is still big enough to be partitionated
                        {
                            Asteriod newAsteroid_1(8, asteroidArr[i].getScale() - 1, asteroidArr[i].getPosition(), gameWidth, gameHeight);
                            asteroidArr.push_back(newAsteroid_1);
                            
                            Asteriod newAsteroid_2(8, asteroidArr[i].getScale() - 1, asteroidArr[i].getPosition(), gameWidth, gameHeight);
                            asteroidArr.push_back(newAsteroid_2);
                        }
                        asteroidArr.erase(asteroidArr.begin() + i);
                        asteroidHitSound.play();
                    }
                }
            }

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
        if (myid == 0) {
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
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}