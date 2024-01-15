// road.cpp
#include "road.h"

#include <iostream>
#include <vector>

#include "movable.h"  // Incluez le fichier d'en-tête de la classe Movable
// #define NODE_RADIUS 10
extern double NODE_RADIUS;
extern double ROAD_OFFSET;
extern std::vector<double> TIME_COUNTER_VEC;
extern double SPEED_TRIGGER;
extern double TIME;

Road::Road(Node* start, Node* end, double speedLimit)
    : start{start}, end{end}, speedLimit{speedLimit}, aiFlowCount0{0}, aiFlowCount1{0}, aiWaitDuration{0}, aiWaitDurationCumulative{0} {
    length = start->position.getLength(end->position);

    Point u = end->position.minus(start->position);
    u = u.multiply(1 / u.getLength());
    Point v = u.ortho();

    posStart = start->position;
    posEnd = end->position;
    roadLen = length - 2 * NODE_RADIUS;

    // posStart.x += 5*u.x + 2*v.x;
    // posStart.y += 5*u.y + 2*v.y;
    // posEnd.x += -5*u.x
    Point u5 = u.multiply(NODE_RADIUS);
    Point v2 = v.multiply(ROAD_OFFSET);
    posStart = posStart.add(u5).add(v2);
    posEnd = posEnd.minus(u5).add(v2);

    start->addRoadOut(this);
    end->addRoadIn(this);
    static int s_id{0};
    id = s_id;
    s_id++;
}
// TODO add ia_flow_count
bool Road::spawnMovable(Movable* movable) {
    // movables.push_back(movable);
    size_t n = movables.size();

    if (n == 0) {
        movables.push_front(movable);
        movable->setRoad(this);
        return true;
    }

    double movEnd = movable->pos + movable->size;
    double movStart = movable->pos - movable->size;
    if (movEnd < movables.front()->minValue()) {
        movables.push_front(movable);
        movable->setRoad(this);
        return true;
    }
    if (movStart > movables.back()->maxValue()) {
        movables.push_back(movable);
        movable->setRoad(this);
        return true;
    }
    auto previous = movables.begin();
    auto it = std::next(previous);
    for (int i = 1; i < n - 1; i++, it++) {
        // increment it as well
        Movable* movPrev = *previous;
        Movable* mov = *it;
        if (movPrev->maxValue() < movable->minValue() && mov->minValue() > movable->maxValue()) {
            movables.insert(it, movable);
            movable->setRoad(this);
            return true;
        }
        previous = it;
    }
    return false;
}

bool Road::addMovable(Movable* movable) {
    if (spawnMovable(movable)) {
        aiFlowCount0++;
        return true;
    }
    return false;
}

void Road::removeMovable(Movable* movable) {
    despawnMovable(movable);
    aiFlowCount1 += 1;
}

void Road::despawnMovable(Movable* movable) {
    for (auto it = movables.begin(); it != movables.end(); it++) {
        Movable* mov = *it;
        if (mov == movable) {
            movables.erase(it);
            break;
        }
    }
}

int Road::getId() const {
    return id;
}

const std::list<Movable*>& Road::getMovables() const {
    return movables;
}

double Road::getSpeedLimit() {
    return speedLimit;
}

void Road::update() {
    Movable* previous = nullptr;
    // std::cout << movables.size() << std::endl;
    bool timeCounter = blockTraffic;
    size_t index = 0;
    for (auto it = movables.rbegin(); it != movables.rend(); ++it) {
        Movable* mov = *it;
        // std::cout << "r : " << mov->nxt_pos << std::endl;

        if (mov->inner_timer > 0) {
            previous = mov;
            timeCounter = false;
            continue;
        }
        mov->nxt_pos = mov->nextPos();
        // std::cout << "r : " << mov->nxt_pos << std::endl;
        if (previous == nullptr) {
            mov->handleFirstMovable();
            if(mov->road_goal == this) {
                timeCounter = false;
            }
        } else {
            collisionDetection(previous, mov);
        }

        mov->handleRoadTarget();
        if(mov->speed > SPEED_TRIGGER) {
            timeCounter = false;
        }
        if (timeCounter) {
            if(previous == nullptr) {
                aiWaitDuration += TIME;
            }
            
            if(index >= TIME_COUNTER_VEC.size()) {
                aiWaitDurationCumulative += TIME_COUNTER_VEC.back();
            } else {
                aiWaitDurationCumulative += TIME_COUNTER_VEC[index++];
            }

        }
        previous = mov;
    }
}

void Road::collisionDetection(Movable* previous, Movable* nxt) {
    if (nxt->nxt_pos + 2 * nxt->size > previous->nxt_pos - 2 * previous->size) {
        nxt->handlePossibleCollision(*previous);
    } else {
        nxt->noPossibleCollision(*previous);
    }
}

std::vector<double> Road::getPosEnd() {
    std::vector<double> res = {posEnd.x, posEnd.y};
    return res;
}

std::vector<double> Road::getPosStart() {
    std::vector<double> res = {posStart.x, posStart.y};
    return res;
}

bool Road::getBlockTraffic() {
    return blockTraffic;
}

void Road::setBlockTraffic(bool b) {
    blockTraffic = b;
}

double Road::getRoadLen() {
    return roadLen;
}

int Road::getAiFlowCount0() {
    return aiFlowCount0;
}

int Road::getAiFlowCount1() {
    return aiFlowCount1;
}

double Road::getAiWaitDuration() {
    return aiWaitDuration;
}
double Road::getAiWaitDurationCumulative() {
    return aiWaitDurationCumulative;
}
void Road::resetAiMetrics() {
    aiFlowCount0 = 0;
    aiFlowCount1 = 0;
    aiWaitDuration = 0;
    aiWaitDurationCumulative = 0;
}