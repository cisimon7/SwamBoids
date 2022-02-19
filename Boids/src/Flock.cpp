//
// Created by Ryan Strauss on 12/10/19.
//

#include "../include/Flock.h"
#include "../include/KDTree.h"

Flock::Flock() = default;

Flock::Flock(const Flock &other) {
    for (const Boid &b: other.boids) boids.emplace_back(b);
}

Flock::~Flock() = default;

Boid Flock::operator[](int i) const {
    return boids[i];
}

void Flock::add(const Boid &boid) {
    boids.emplace_back(boid);
}

void Flock::clear() {
    boids.clear();
}

void Flock::update(float window_width, float window_height, int num_threads) {
    KDTree tree(window_width, window_height);
    for (Boid &b: boids) tree.insert(&b);

    /* An array of boid arrays representing the boids close to a particular boid. */
    std::vector<Boid *> search_results[boids.size()];

#pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1)
    for (int i = 0; i < boids.size(); ++i) {
        Boid &b = boids[i];
        search_results[i] = tree.search(&b, b.perception);
    }

#pragma omp parallel for schedule(dynamic) num_threads(num_threads) if (num_threads > 1)
    for (int i = 0; i < boids.size(); ++i)
        /* Takes the array of boids within its radius and makes decision on how it moves next step*/
        boids[i].update(search_results[i]);
}

int Flock::size() const {
    return boids.size();
}
