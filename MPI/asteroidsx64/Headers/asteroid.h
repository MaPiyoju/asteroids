#pragma once
#include <SFML/Graphics.hpp>
#include "bulllet.h"
#include "ship.h"

class Asteriod
{
public:
    Asteriod(int points, int initScale, sf::Vector2f pos, int gameWidth, int gameHeigth)
    {
        asteroid.setPointCount(points);
        for (int i = 0; i < points; i++)
        {
            int x = 0;
            int y = 0;
            // Generate polygon by sectors
            if (i > 1 && i < 6)
            {
                x = rand() % (20 - 10 + 1) + 10;
            }
            else
            {
                x = rand() % 10;
            }
            if (i < 4)
            {
                y = rand() % 10;
            }
            else
            {
                y = rand() % (20 - 10 + 1) + 10;
            }
            asteroid.setPoint(i, sf::Vector2f(x * initScale, y * initScale));
        }
        asteroid.setOutlineColor(sf::Color::White);
        asteroid.setOutlineThickness(1);
        asteroid.setFillColor(sf::Color::Black);
        asteroid.setOrigin(asteroid.getLocalBounds().width / 2, asteroid.getLocalBounds().height / 2);
        asteroid.setPosition(pos);
        gameW = gameWidth;
        gameH = gameHeigth;
        scale = initScale;

        dir = 1;
        int dirRand = rand() % 1;
        if (dirRand == 0)
        {
            dir = -1;
        }
    }

    void move(sf::Vector2f dir)
    {
        asteroid.move(dir);

        // Check X positions
        if (asteroid.getPosition().x + asteroid.getLocalBounds().width < 0)
        {
            asteroid.setPosition(gameW, asteroid.getPosition().y);
        }
        if (asteroid.getPosition().x - asteroid.getLocalBounds().width / 2 > gameW)
        {
            asteroid.setPosition(-asteroid.getLocalBounds().width, asteroid.getPosition().y);
        }

        // Check Y positions
        if (asteroid.getPosition().y + asteroid.getLocalBounds().height < 0)
        {
            asteroid.setPosition(asteroid.getPosition().x, gameH);
        }
        if (asteroid.getPosition().y - asteroid.getLocalBounds().height / 2 > gameH)
        {
            asteroid.setPosition(asteroid.getPosition().x, -asteroid.getLocalBounds().height);
        }
    }

    void changePosition(sf::Vector2f pos) {
        asteroid.setPosition(pos);
    }

    void rotate(float angle)
    {
        asteroid.rotate(angle * dir);
    }

    bool bulletCollision(Bullet bullet)
    {
        if (bullet.getRight() >= asteroid.getPosition().x &&
            bullet.getLeft() <= (asteroid.getPosition().x + asteroid.getLocalBounds().width) &&
            bullet.getTop() >= asteroid.getPosition().y &&
            bullet.getBottom() <= (asteroid.getPosition().y + asteroid.getLocalBounds().height))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool shipCollision(Ship ship)
    {
        if (ship.getRight() >= asteroid.getPosition().x &&
            ship.getLeft() <= (asteroid.getPosition().x + asteroid.getLocalBounds().width) &&
            ship.getTop() >= asteroid.getPosition().y &&
            ship.getBottom() <= (asteroid.getPosition().y + asteroid.getLocalBounds().height))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    int getScale()
    {
        return scale;
    }

    sf::Vector2f getPosition()
    {
        return asteroid.getPosition();
    }

    float getRotation()
    {
        return asteroid.getRotation();
    }

    sf::FloatRect getBounds()
    {
        return asteroid.getLocalBounds();
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(asteroid);
    }

private:
    sf::ConvexShape asteroid;
    int gameW;
    int gameH;
    int scale;
    int dir;
};