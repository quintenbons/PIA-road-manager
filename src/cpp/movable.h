// movable.h
#ifndef MOVABLE_H
#define MOVABLE_H

#include "point.h"
#include "vector"
#include "road.h"
#include "node.h"

struct resNodePos {
    double pos;
    double speed;
    Point p;
};

class Road;
class Node;

class Movable {
   public:
    Movable(double speed, double acceleration, double pos, double size, double spawn_tick, double TIME);

    void updatePosition();
    double nextPos();
    double nextSpeed();
    auto nextPosition();
    void handleRoadTarget();
    void handleFirstMovable();
    void handlePossibleCollision(Movable& other);
    void noPossibleCollision(Movable& other);
    bool updateRoad();
    void notifyNodeCollision();
    void notifyNodePriority();
    resNodePos nextNodePosition();
    void updateNode();
    bool update();
    void setRoad(Road* road);
    void stop();
    void setRoadPath(Road* arrival);
    void setRoadGoal(Road* arrival, double pos);
    Road* findNextRoad(Node* next_node);

    double maxValue();
    double minValue();
    double getScore(double currentTick);

    double getPos();

    std::vector<double> toCoordXY();
    int getId() const;

    double const TIME;

    Road* road;
    Node* node;
    int lane;
    double speed;
    double pos;
    double nxt_pos;
    double nxt_speed;

    double acceleration;
    double current_acceleration;
    // double latency;
    Point node_pos;
    Point node_dir;

    bool node_mov;
    double node_len;
    double size;
    double spawn_tick;

    std::vector<Node*> path;

    Road* road_goal;
    double pos_goal;
    int inner_timer;
    int id;

   private:
};
#endif