#pragma once
#include <SFML/Graphics.hpp>

class Bullet
{
public:
    Bullet(float radius)
    {
        bullet.setRadius(radius);
    }

    void fire(int speed)
    {
        bullet.move(speed, 0);
    }

    int getRight()
    {
        return bullet.getPosition().x + (bullet.getRadius() * 2);
    }

    int getLeft()
    {
        return bullet.getPosition().x;
    }

    int getTop()
    {
        return bullet.getPosition().y;
    }

    int getBottom()
    {
        return bullet.getPosition().y + (bullet.getRadius() * 2);
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(bullet);
    }

private:
    sf::CircleShape bullet;
};