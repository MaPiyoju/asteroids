#pragma once
#include <SFML/Graphics.hpp>

class Ship
{
public:
    Ship(sf::Vector2f position)
    {
        ship.setPointCount(4);
        ship.setPoint(0, sf::Vector2f(0, 0));
        ship.setPoint(1, sf::Vector2f(5, 5));
        ship.setPoint(2, sf::Vector2f(0, 10));
        ship.setPoint(3, sf::Vector2f(20, 5));
        ship.setOutlineColor(sf::Color::White);
        ship.setOutlineThickness(1);
        ship.setPosition(position);
    }

    void move(sf::Vector2f dir)
    {
        ship.move(dir);
    }

    sf::Vector2f getPosition()
    {
        return ship.getPosition();
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(ship);
    }

private:
    sf::ConvexShape ship;
};