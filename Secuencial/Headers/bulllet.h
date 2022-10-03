#pragma once
#include <SFML/Graphics.hpp>

class Bullet
{
public:
    Bullet(float radius, sf::Vector2f initPos, float pi, float rotation)
    {
        bullet.setRadius(radius);
        bullet.setOutlineThickness(3);
        bullet.setOutlineColor(sf::Color::Black);
        bullet.setFillColor(sf::Color::White);
        bullet.setOrigin(radius / 2, radius / 2);
        bullet.setPosition(initPos);

        float bulletAngle = (pi / 180) * (rotation); // Convert bullet's angle to radians

        // Idle movement, since it's space ship will always be moving
        angleX = cos(bulletAngle);
        angleY = sin(bulletAngle);

        collided = false;
    }

    void move(float speed)
    {
        bullet.move(sf::Vector2f(angleX * speed, angleY * speed));
    }

    bool checkBounds(int gameW, int gameH)
    {
        // Check X positions
        if (bullet.getPosition().x + bullet.getLocalBounds().width < 0)
        {
            return true;
        }
        if (bullet.getPosition().x > gameW)
        {
            return true;
        }

        // Check Y positions
        if (bullet.getPosition().y + bullet.getLocalBounds().height < 0)
        {
            return true;
        }
        if (bullet.getPosition().y > gameH)
        {
            return true;
        }
        return false;
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

    void collide()
    {
        collided = true;
    }

    bool getCollided()
    {
        return collided;
    }

    void draw(sf::RenderWindow &window)
    {
        window.draw(bullet);
    }

private:
    sf::CircleShape bullet;
    float angleX;
    float angleY;
    bool collided;
};