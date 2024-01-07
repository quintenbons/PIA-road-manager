#include "node.h"
#include <iostream>

extern int MAX_MOVABLES_IN_NODE;

Node::Node(double x, double y) : position(x, y) {
    static int s_id{0};
    id = s_id;
    s_id++;
}

bool circleCollision(Point p1, Point p2, double size1, double size2) {
    double dist = p1.getLength(p2);
    if (dist < (size1 + size2)) {
        return true;
    }
    return false;
}

void Node::update(py::object obj, double timer) {
    if(!obj.is_none()) {
        obj.attr("update")(timer);
    }
    int n = movables.size();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            Movable* movable1 = movables[i];
            Movable* movable2 = movables[j];

            auto res1 = movable1->nextNodePosition();
            auto res2 = movable2->nextNodePosition();

            Point A = movable1->node_pos;
            Point B = movable2->node_pos;
            Point O = B.minus(A);
            Point U = res1.p.minus(A);
            Point V = res2.p.minus(B);
            Point N = U.ortho();
            double sca = N.scalaire(O);

            Movable* priorityMov = nullptr;
            Movable* stopMov = nullptr;

            if (sca > 0) {
                priorityMov = movable2;
                stopMov = movable1;
            } else {
                priorityMov = movable1;
                stopMov = movable2;
            }

            if (circleCollision(res1.p, res2.p, movable1->size, movable2->size)) {
                stopMov->notifyNodeCollision();
                priorityMov->notifyNodePriority();
            }

            double det = V.x * U.y - V.y * U.x;
            double t = V.x * O.y - O.x * V.y;

            if (det != 0.0) {
                t /= det;
                Point Ut = U.multiply(t);
                Point P = A.add(Ut);

                Point AP = P.minus(A);
                Point BP = P.minus(B);

                double sca1 = AP.scalaire(U);
                double sca2 = BP.scalaire(V);

                if (sca1 >= 0 && sca2 >= 0) {
                    if (AP.getLength() < U.getLength() and BP.getLength() < V.getLength()) {
                        stopMov->notifyNodeCollision();
                        priorityMov->notifyNodePriority();
                    }
                }
            }
        }
    }
}
// #define MAX_MOVABLES_IN_NODE 5
bool Node::positionAvailable(Point& pos, double size) {
    if (movables.size() > MAX_MOVABLES_IN_NODE) {
        return false;
    }
    for (Movable* m : movables) {

        if (circleCollision(m->node_pos, pos, m->size, size)) {
            return false;
        }
    }
    return true;
}

void Node::addRoadIn(Road* road) {
    roadIn.push_back(road);
}

void Node::addRoadOut(Road* road) {
    roadOut.push_back(road);
}

Road* Node::roadTo(Node* node) {
    double dist = std::numeric_limits<double>::infinity();
    Road* road = nullptr;
    for (Road* r : roadOut) {

        if (r->end == node && r->length < dist) {
            dist = r->length;
            road = r;
        }
    }

    assert(road != nullptr);
    return road;
}

void Node::setPath(std::unordered_map<Node*, Node*> p) {
    for (auto& couple : p) {
        // paths.insert(couple.first, couple.second);
        paths[couple.first] = couple.second;
        // paths.operator[couple.first] = couple.second;
    }
}

void Node::constructPath(Node* n1, Node* n2) {
    paths[n1] = n2;
}

std::vector<Node*> Node::findPath(Node* other) {
    std::vector<Node*> path;
    Node* current = other;
    while (current != this) {
        path.push_back(current);
        current = paths[current];
    }
    path.push_back(this);

    return path;
}

double Node::getX() {
    return this->position.x;
}

double Node::getY() {
    return this->position.y;
}

void Node::setPosition(double x, double y) {
    this->position = Point(x, y);
}

std::vector<Road*> Node::getRoadIn() {
    return roadIn;
}
std::vector<Road*> Node::getRoadOut() {
    return roadOut;
}

int Node::getId() {
    return id;
}