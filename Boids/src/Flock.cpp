/*
 * Created by Ryan Strauss on 12/9/19.
 * Extended by Simon Idoko on 20/02/22
 * */

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


//TODO(How to parallelize this in python and at the same time make use of the KDTree structure)
void Flock::update(float window_width, float window_height, int num_threads) {
    /*
     * TODO(
     *    This Builds a KDTree structure every render step
     *    This is because at each time step, the positions of the boids have changed and hence need for new KDTree
     *    But can this be improved? Is there a better way?
     * )
     * */
    KDTree tree(window_width, window_height); /*Creates a new KDTree from current positions of boids*/
    for (Boid &b: boids) tree.insert(&b); /*Arranges the boids into a KDTree structure for faster distance searching*/

    /* An array of boid array representing the boids close to a particular boid. */
    // TODO(I would use mapping in kotlin in this case to find for each boid, its circle of reach)
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

const std::vector<Boid> &Flock::getBoids() const {
    return boids;
}

void Flock::setBoids(const std::vector<Boid> &boids_) {
    Flock::boids = boids_;
}
