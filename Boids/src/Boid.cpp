/*
 * Created by Ryan Strauss on 12/9/19.
 * Extended by Simon Idoko on 20/02/22
 * */

#include <cmath>
#include "../include/Boid.h"

Boid::Boid(int boid_id, float x, float y, float max_width, float max_height, float max_speed, float max_force,
           float acceleration_scale, float cohesion_weight, float alignment_weight, float separation_weight,
           float perception, float separation_distance, float noise_scale, bool is_predator) {
    position = Vector2D{x, y};
    velocity = (Vector2D::random() - 0.5) * max_speed * 2;
    acceleration = Vector2D{};

    this->boid_id = boid_id;
    this->max_width = max_width;
    this->max_height = max_height;
    this->max_speed = max_speed;
    this->max_force = max_force;
    this->acceleration_scale = acceleration_scale;
    this->cohesion_weight = cohesion_weight;
    this->alignment_weight = alignment_weight;
    this->separation_weight = separation_weight;
    this->perception = perception;
    this->separation_distance = separation_distance;
    this->noise_scale = noise_scale;
    this->is_predator = is_predator;

    if (is_predator) {
        this->max_speed *= PREDATOR_SPEED_BOOST;
        this->perception *= PREDATOR_PERCEPTION_BOOST;
    }
}

Boid::Boid(const Boid &other) {
    boid_id = other.boid_id;
    position = other.position;
    velocity = other.velocity;
    acceleration = other.acceleration;
    max_width = other.max_width;
    max_height = other.max_height;
    max_speed = other.max_speed;
    max_force = other.max_force;
    acceleration_scale = other.acceleration_scale;
    cohesion_weight = other.cohesion_weight;
    alignment_weight = other.alignment_weight;
    separation_weight = other.separation_weight;
    perception = other.perception;
    separation_distance = other.separation_distance;
    noise_scale = other.noise_scale;
    is_predator = other.is_predator;
}

Boid::~Boid() = default;

Boid &Boid::operator=(const Boid &other) = default;

/**
 * Unit vector in direction of average direction (direction of each boid from vec position) of all boids */
Vector2D Boid::alignment(const std::vector<Boid *> &boids) const {
    /**
     * return zero vector if no boids in reachability radius
     * return unit vector in direction of perceived velocity
     * perceived velocity is the average in each direction of the velocity vectors of all boids in reachability radius*/
    Vector2D perceived_velocity; /** Default as zero vector */
    int n = 0;

    for (const Boid *b: boids) {
        if (this != b) {
            if (b->is_predator)
                return Vector2D{};

            perceived_velocity += b->velocity; // Velocity a better measure of direction
            ++n;
        }
    }

    if (n == 0)
        return Vector2D{};

    perceived_velocity /= n;
    Vector2D steer = perceived_velocity - velocity;
    return steer.normalize();
}

/**
 * A unit vector from boid position to center of neighbor boids */
Vector2D Boid::cohesion(const std::vector<Boid *> &boids) const {
    Vector2D perceived_center; /** Default as zero vector */
    int n = 0;

    /**
     * If there is a predator within the reachability radius, return zero vector */
    for (const Boid *b: boids) {
        if (this != b) {
            if (b->is_predator)
                return Vector2D{};

            perceived_center += b->position;
            ++n;
        }
    }

    /**
     * No boids within reachability radius, return zero vector */
    if (n == 0)
        return Vector2D{};

    /** average to find perceived centre */
    perceived_center /= n;
    Vector2D steer = perceived_center - position;
    return steer.normalize(); /** returns unit vector pointing in direction of perceived centre */
}

Vector2D Boid::separation(const std::vector<Boid *> &boids) const {
    Vector2D c; /** Default as zero vector */

    for (const Boid *b: boids) {
        if (this != b) {
            if (!is_predator && b->is_predator) {
                /**
                 * A normal boid detects a predator within its reachability radius, boid should run away far as possible
                 * return vector in direction opposite to predator with magnitude as large as possible
                 * This would be limited by the presence of other boids of course, so full effect not see on render*/
                return (b->position - position).normalize() * -PREDATOR_ESCAPE_FACTOR;
            } else if (is_predator == b->is_predator &&
                       position.toroidal_distance2(b->position, max_width, max_height) <
                       separation_distance * separation_distance) {
                /**
                 * A boid/predator detects it's fellow boid/predator and their distance is less than the separation distance
                 * sum difference in their position and return its inverse. */
                c -= b->position - position; /*This is same as summing all and then negating. Inverse proportionality*/
            }
        }
    }

    return c.normalize();
}

void Boid::update(const std::vector<Boid *> &boids) {
    // Apply each rule, get resulting forces, and weight them
    Vector2D alignment_update = alignment(boids) * alignment_weight; /** weighted step on unit velocity vector pointing to vel center*/
    Vector2D cohesion_update = cohesion(boids) * cohesion_weight; /** weighted step on unit position vector pointing to pos center */
    Vector2D separation_update = separation(boids) * separation_weight; /** weighted unit vector in direction to maximize distance between all neighboring boids*/
    // Apply the weighted forces to this boid
    acceleration += alignment_update + cohesion_update + separation_update;
    // Scale the acceleration then use it to update the velocity
    if (is_predator)
        acceleration *= PREDATOR_ACCELERATION_BOOST;
    acceleration *= acceleration_scale;
    acceleration.limit(max_force);
    velocity += acceleration;
    if (noise_scale != 0)
        velocity += (Vector2D::random() - 0.5) * noise_scale; /** measuring velocity in reality is not perfect */
    // Limit the velocity so the boids don't get too fast
    velocity.limit(max_speed);
    // Then update the position based on the velocity
    position += velocity;
    // Set the acceleration to zero before the next update
    acceleration = 0;
    // If boid leaves the screen, update position so the boid wraps around
    if (position.x < 0) position.x += max_width;
    if (position.y < 0) position.y += max_height;
    if (position.x >= max_width) position.x -= max_width;
    if (position.y >= max_height) position.y -= max_height;
}

float Boid::angle() const {
    return (float) (atan2(velocity.x, -velocity.y) * 180 / M_PI);
}
