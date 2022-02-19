//
// Created by Ryan Strauss on 6/30/20.
//

#include <cmath>
#include "../include/KDTree.h"


Node::Node(Boid *boid, bool vertical, const Node::NodePtr &left, const Node::NodePtr &right) {
    this->boid = boid;
    this->vertical = vertical;
    this->left = left;
    this->right = right;
}

Node::~Node() = default;

/* A Leaf is a node without left and right Node */
bool Node::isLeaf() const {
    return this->left == nullptr && this->right == nullptr;
}

KDTree::KDTree(float width, float height) {
    this->root = nullptr;
    this->width = width;
    this->height = height;
}

/* Takes a boid and makes a Leaf Node to be added to the KDTree at depth K */
/*TODO(I wonder if this is a correct implementation of a KDTree Structure)*/
Node::NodePtr KDTree::insert_(const Node::NodePtr &node, Boid *boid, bool vertical) {

    /* Function travers down the KDTree until it gets to depth K signified by [node] being [nullptr],
     * A Leaf Node is created at this point and added to the KDTree via recursion. */
    if (node == nullptr) {
        return std::make_shared<Node>(boid, vertical, nullptr, nullptr); /* Leaf Node */
    }

    if ((node->vertical && boid->position.x < node->boid->position.x) ||
        (!node->vertical && boid->position.y < node->boid->position.y)) {
        node->left = insert_(node->left, boid, !node->vertical);
    } else {
        node->right = insert_(node->right, boid, !node->vertical);
    }

    /* Returns the Node after appending Leaf Node at depth K. */
    return node;
}

/* Takes a boid and adds it to root KDTree at depth K */
void KDTree::insert(Boid *boid) {
    root = insert_(root, boid, true);
}

/* Euclidean distance between [boid] and boid at [node]. */
double distance2(const Boid *boid, const Node::NodePtr &node) {
    return std::pow((boid->position.x - node->boid->position.x), 2) +
           std::pow((boid->position.y - node->boid->position.y), 2);
}

/* Finds all the boids within a given radius */
void KDTree::search_(Boid *query, double radius, const Node::NodePtr &node, std::vector<Boid *> &results) const {
    double w = query->position.toroidal_distance2(node->boid->position, this->width, this->height);

    /* If node is a LEAF and node boid is within distance, add to result and return */
    if (node->isLeaf()) {
        if (w < radius * radius) {
            results.push_back(node->boid);
        }
        return;
    }

    /* If node not a LEAF and node boid within distance, add to result and move search branches. */
    if (w < radius * radius)
        results.push_back(node->boid);

    double qx = query->position.x;
    double nx = node->boid->position.x;
    double qy = query->position.y;
    double ny = node->boid->position.y;

    if (node->vertical) {
        if (qx < nx) {
            /* (qx - r) <= nx <= (qx + r) */
            //TODO(nx is already greater than qx, hence [qx - radius <= nx] will always be true)
            if (qx - radius <= nx && node->left != nullptr) {
                search_(query, radius, node->left, results);
            }
            if (qx + radius > nx && node->right != nullptr) {
                search_(query, radius, node->right, results);
                return;
            }
        } else {
            //TODO(already qx >= nx, so qx + radius is definitely nx)
            if (qx + radius > nx && node->right != nullptr) {
                search_(query, radius, node->right, results);
            }
            if (qx - radius <= nx && node->left != nullptr) {
                search_(query, radius, node->left, results);
                return;
            }
        }
    } else {
        if (qy < ny) {
            if (qy - radius <= ny && node->left != nullptr) {
                search_(query, radius, node->left, results);
            }
            if (qy + radius > ny && node->right != nullptr) {
                search_(query, radius, node->right, results);
                return;
            }
        } else {
            if (qy + radius > ny && node->right != nullptr) {
                search_(query, radius, node->right, results);
            }
            if (qy - radius <= ny && node->left != nullptr) {
                search_(query, radius, node->left, results);
                return;
            }
        }
    }
}

std::vector<Boid *> KDTree::search(Boid *query, double radius) const {
    std::vector<Boid *> results;
    search_(query, radius, root, results);
    return results;
}
