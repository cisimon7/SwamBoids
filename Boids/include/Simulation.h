/*
 * Created by Ryan Strauss on 12/9/19.
 * Extended by Simon Idoko on 20/02/22
 * */

#ifndef BOIDS_SIMULATION_H
#define BOIDS_SIMULATION_H

#include <chrono>
#include <vector>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "Flock.h"

class Simulation {
private:
    sf::RenderWindow window;
    sf::VideoMode step_desktop;
    int window_width, window_height;
    Flock flock;
    std::vector<sf::CircleShape> shapes;

    bool light_scheme, fullscreen;
    float max_speed, max_force;
    float acceleration_scale;
    float perception, separation_distance;
    float cohesion_weight, alignment_weight, separation_weight;
    float noise_scale;
    float boid_size;
    int num_threads;
    int frame_rate;

    void render(const std::function<void()> &on_frame, bool ext_update = false);

    void add_boid(float x, float y, bool is_predator = false, bool with_shape = true);

    bool handle_input();

public:
    constexpr static float DEFAULT_BOID_SIZE = 4;
    constexpr static int DEFAULT_WINDOW_WIDTH = 1500;
    constexpr static int DEFAULT_WINDOW_HEIGHT = 900;
    constexpr static int DEFAULT_FLOCK_SIZE = 50;

    constexpr static float DEFAULT_MAX_SPEED = 6;
    constexpr static float DEFAULT_MAX_FORCE = 1;
    constexpr static float DEFAULT_ALIGNMENT_WEIGHT = 0.65;
    constexpr static float DEFAULT_COHESION_WEIGHT = 0.75;
    constexpr static float DEFAULT_SEPARATION_WEIGHT = 4.5;
    constexpr static float DEFAULT_ACCELERATION_SCALE = 0.3;
    constexpr static float DEFAULT_PERCEPTION = 100;
    constexpr static float DEFAULT_SEPARATION_DISTANCE = 20;

    constexpr static float DEFAULT_NOISE_SCALE = 0;

    const Flock &getFlock() const;

    void setFlock(const Flock &flock_);

    const std::vector<sf::CircleShape> &getShapes() const;

    void setShapes(const std::vector<sf::CircleShape> &shapes_);

    std::function<void(float)> frame_update = [](float a) {};

    Simulation(int window_width, int window_height, float boid_size, float max_speed, float max_force,
               float alignment_weight,
               float cohesion_weight, float separation_weight, float acceleration_scale, float perception,
               float separation_distance, float noise_scale, bool fullscreen = false, bool light_scheme = false,
               int num_threads = -1, int FRAME_RATE = 60);

    ~Simulation();

    void run(int flock_size, int pred_size = 0, const std::function<void()> &on_frame = []() {}, bool ext_update = false);

    std::vector<double> benchmark(int flock_size, int num_steps);

    void step_run(int flock_size, int pred_size = 0, const std::function<void()> &on_frame = []() {}, int delay_ms = 500);

    float static get_random_float();
};


#endif //BOIDS_SIMULATION_H
