#ifndef ROAD_H
#define ROAD_H

#include <list>
#include "point.h"
#include "node.h"
#include "movable.h"

class Road {
public:
    Road(Node* start, Node* end, double speedLimit);
    bool addMovable(Movable* movable);
    bool spawnMovable(Movable* movable);
    int getId() const;
    const std::list<Movable*>& getMovables() const;
    double getSpeedLimit();
    void removeMovable(Movable* mov);
    void despawnMovable(Movable* mov);
    void update();
    void collisionDetection(Movable* previous, Movable* nxt);
    std::vector<double> getPosEnd();
    std::vector<double> getPosStart();
    bool getBlockTraffic();
    void setBlockTraffic(bool b);
    double getRoadLen();
    Node* start;
    Node* end;
    Point posStart;
    Point posEnd;
    double roadLen;
    bool bidirectional;
    double length;
    double speedLimit;

    bool blockTraffic;
    int aiFlowCount0;
    int aiFlowCount1;

    int id;
    // std::vector<BinarySearchTree*> lanes;
private:
    std::list<Movable*> movables;  // Liste d'objets Movable
};

#endif