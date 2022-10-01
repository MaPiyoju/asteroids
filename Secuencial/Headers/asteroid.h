#pragma once
#include <SFML/Graphics.hpp>
#include "bulllet.h"

class Asteriod
{
public:
    Asteriod(int points, int scale)
    {
        asteroid.setPointCount(points);
        asteroid.setPoint(0, sf::Vector2f(0, 0));
        for (int i = 1; i < points; i++)
        {
            int x = rand() % 20;
            int y = rand() % 20;
            asteroid.setPoint(i, sf::Vector2f(x * scale, y * scale));
        }
        asteroid.setOutlineColor(sf::Color::White);
        asteroid.setOutlineThickness(1);
        asteroid.setPosition(10, 10);
    }

    void move(sf::Vector2f dir, int gameW, int gameH)
    {
        asteroid.move(dir);

        // Check X positions
        if (asteroid.getPosition().x + asteroid.getLocalBounds().width < 0)
        {
            asteroid.setPosition(gameW, asteroid.getPosition().y);
        }
        if (asteroid.getPosition().x > gameW)
        {
            asteroid.setPosition(-asteroid.getLocalBounds().width, asteroid.getPosition().y);
        }

        // Check Y positions
        if (asteroid.getPosition().y + asteroid.getLocalBounds().height < 0)
        {
            asteroid.setPosition(asteroid.getPosition().x, gameH);
        }
        if (asteroid.getPosition().y > gameH)
        {
            asteroid.setPosition(asteroid.getPosition().x, -asteroid.getLocalBounds().height);
        }
    }

    void collision(Bullet bullet)
    {
        if (bullet.getRight() >= asteroid.getPosition().x &&
            bullet.getLeft() <= (asteroid.getPosition().x + asteroid.getLocalBounds().width) &&
            bullet.getTop() >= asteroid.getPosition().y &&
            bullet.getBottom() <= (asteroid.getPosition().y + asteroid.getLocalBounds().height))
        {
            asteroid.setPosition(0, 0);
        }
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(asteroid);
    }

private:
    sf::ConvexShape asteroid;
};