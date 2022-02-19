//
// Created by Ryan Strauss on 6/30/20.
//

#ifndef BOIDS_KDTREE_H
#define BOIDS_KDTREE_H

#include <vector>
#include <memory>
#include "Boid.h"

/* A Node holds a boid */
class Node {
public:
    using NodePtr = std::shared_ptr<Node>;

    NodePtr left, right;
    Boid *boid;
    bool vertical; /* Alternating Axis */

    Node(Boid *boid, bool vertical, const NodePtr &left, const NodePtr &right);

    ~Node();

    bool isLeaf() const;
};

/* KDTree holding Boids */
class KDTree {
private:
    float width, height;

    Node::NodePtr root;

    /* Builds the Boid KDTree Structure */
    Node::NodePtr insert(const Node::NodePtr &node, Boid *boid, bool vertical);

    /**
     * Search for boids [query] to a given [node] within a given radius and store in [results]  */
    void search(Boid *query, double radius, const Node::NodePtr &node, std::vector<Boid *> &results) const;

public:
    KDTree(float width, float height);

    void insert(Boid *boid);

    std::vector<Boid *> search(Boid *query, double radius) const;
};


#endif //BOIDS_KDTREE_H
