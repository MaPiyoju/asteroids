#pragma once
#include <SFML/Graphics.hpp>

class Ship
{
public:
    Ship(sf::Vector2f position, int gameWidth, int gameHeight)
    {
        // Ship body
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

        // Propultion body
        propultion.setPointCount(4);
        propultion.setPoint(0, sf::Vector2f(5, 5));
        propultion.setPoint(1, sf::Vector2f(0, 10));
        propultion.setPoint(2, sf::Vector2f(-6, 5));
        propultion.setPoint(3, sf::Vector2f(0, 0));
        propultion.setOutlineColor(sf::Color::White);
        propultion.setFillColor(sf::Color::Black);
        propultion.setOutlineThickness(1);
        propultion.setPosition(position);
        propultion.setOrigin(ship.getLocalBounds().width / 2, propultion.getLocalBounds().height / 2);
        propultionState = false;
    }

    void move(sf::Vector2f dir)
    {
        ship.move(dir);

        /* Propultion */
        propultion.move(dir);

        // Check X positions
        if (ship.getPosition().x + ship.getLocalBounds().width < 0)
        {
            ship.setPosition(gameW, ship.getPosition().y);
            propultion.setPosition(ship.getPosition().x, ship.getPosition().y);
        }
        if (ship.getPosition().x > gameW)
        {
            ship.setPosition(-ship.getLocalBounds().width, ship.getPosition().y);
            propultion.setPosition(ship.getPosition().x, ship.getPosition().y);
        }

        // Check Y positions
        if (ship.getPosition().y + ship.getLocalBounds().height < 0)
        {
            ship.setPosition(ship.getPosition().x, gameH);
            propultion.setPosition(ship.getPosition().x, ship.getPosition().y);
        }
        if (ship.getPosition().y > gameH)
        {
            ship.setPosition(ship.getPosition().x, -ship.getLocalBounds().height);
            propultion.setPosition(ship.getPosition().x, ship.getPosition().y);
        }
    }

    void reset()
    {
        ship.setPosition(gameW / 2, gameH / 2);
        propultion.setPosition(gameW / 2, gameH / 2);
    }

    void rotate(float angle)
    {
        ship.rotate(angle);
        propultion.rotate(angle);
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

    void switchPropultion()
    {
        if (propultionState == true)
        {
            propultion.setPoint(0, sf::Vector2f(5, 5));
            propultion.setPoint(1, sf::Vector2f(0, 10));
            propultion.setPoint(2, sf::Vector2f(-6, 5));
            propultion.setPoint(3, sf::Vector2f(0, 0));
        }
        else
        {
            propultion.setPoint(0, sf::Vector2f(5, 5));
            propultion.setPoint(1, sf::Vector2f(0, 7));
            propultion.setPoint(2, sf::Vector2f(-10, 5));
            propultion.setPoint(3, sf::Vector2f(0, 3));
        }
        propultionState = !propultionState;
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(ship);
    }

    void drawPropultion(sf::RenderWindow &window)
    {
        window.draw(propultion);
    }

private:
    sf::ConvexShape ship;
    sf::ConvexShape propultion;
    int gameW;
    int gameH;
    bool propultionState;
};