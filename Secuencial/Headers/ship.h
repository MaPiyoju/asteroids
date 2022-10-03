#pragma once
#include <SFML/Graphics.hpp>

class Ship
{
public:
    Ship(sf::Vector2f position, int gameWidth, int gameHeight)
    {
        ship.setPointCount(4);
        ship.setPoint(0, sf::Vector2f(0, 0));
        ship.setPoint(1, sf::Vector2f(5, 5));
        ship.setPoint(2, sf::Vector2f(0, 10));
        ship.setPoint(3, sf::Vector2f(20, 5));
        ship.setOutlineColor(sf::Color::White);
        ship.setOutlineThickness(1);
        ship.setPosition(position);
        ship.setOrigin(0, ship.getLocalBounds().height / 2);
        gameW = gameWidth;
        gameH = gameHeight;
    }

    void move(sf::Vector2f dir)
    {
        ship.move(dir);
        // Check X positions
        if (ship.getPosition().x + ship.getLocalBounds().width < 0)
        {
            ship.setPosition(gameW, ship.getPosition().y);
        }
        if (ship.getPosition().x > gameW)
        {
            ship.setPosition(-ship.getLocalBounds().width, ship.getPosition().y);
        }

        // Check Y positions
        if (ship.getPosition().y + ship.getLocalBounds().height < 0)
        {
            ship.setPosition(ship.getPosition().x, gameH);
        }
        if (ship.getPosition().y > gameH)
        {
            ship.setPosition(ship.getPosition().x, -ship.getLocalBounds().height);
        }
    }

    void reset()
    {
        ship.setPosition(gameW / 2, gameH / 2);
    }

    void rotate(float angle)
    {
        ship.rotate(angle);
    }

    sf::Vector2f getPosition()
    {
        return ship.getPosition();
    }

    float getRotation()
    {
        return ship.getRotation();
    }

    int getRight()
    {
        return ship.getPosition().x + (ship.getLocalBounds().width);
    }

    int getLeft()
    {
        return ship.getPosition().x;
    }

    int getTop()
    {
        return ship.getPosition().y;
    }

    int getBottom()
    {
        return ship.getPosition().y + (ship.getLocalBounds().height);
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(ship);
    }

private:
    sf::ConvexShape ship;
    int gameW;
    int gameH;
};