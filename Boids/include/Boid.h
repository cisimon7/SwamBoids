/*
 * Created by Ryan Strauss on 12/9/19.
 * Extended by Simon Idoko on 20/02/22
 * */

#ifndef BOIDS_BOID_H
#define BOIDS_BOID_H

#include <vector>
#include "Vector2D.h"

class Boid {
private:
    constexpr static float PREDATOR_ESCAPE_FACTOR = 10000000;
    constexpr static float PREDATOR_SPEED_BOOST = 1.8;
    constexpr static float PREDATOR_PERCEPTION_BOOST = 1.5;
    constexpr static float PREDATOR_ACCELERATION_BOOST = 1.4;

public:
    /**
     * [cohesion_weight]: degree of randomness in movement of boid
     * [alignment_weight]: unity in movement of boids in a group
     * [separation_weight]: distance to maintain between boids in a group
     * [perception]: radius of reachability
     * */
    int boid_id;
    Vector2D position, velocity, acceleration;
    float max_width, max_height;
    float max_speed, max_force;
    float acceleration_scale;
    float cohesion_weight, alignment_weight, separation_weight;
    float perception, separation_distance; /** Perception: Radius which boid can detect other objects*/
    float noise_scale;
    bool is_predator;

    Boid(int boid_id, float x, float y, float max_width, float max_height, float max_speed, float max_force,
         float acceleration_scale, float cohesion_weight, float alignment_weight, float separation_weight,
         float perception, float separation_distance, float noise_scale, bool is_predator = false);

    Boid(const Boid &other);

    ~Boid();

    Boid &operator=(const Boid &other);

    Vector2D alignment(const std::vector<Boid *> &boids) const;

    Vector2D cohesion(const std::vector<Boid *> &boids) const;

    Vector2D separation(const std::vector<Boid *> &boids) const;

    void update(const std::vector<Boid *> &boids);

    float angle() const;

    int getBoidId() const;

};


#endif //BOIDS_BOID_H

/*https://medium.com/swlh/boids-a-simple-way-to-simulate-how-birds-flock-in-processing-69057930c229*/
