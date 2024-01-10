// movable.cpp
#include "movable.h"

#include <iostream>

#include "road.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) < (y) ? (y) : (x))

extern double LEAVING_DIST;
extern double LEAVING_TIME;
extern double TIME;
extern double STOP_SPEED;

Movable::Movable(
    double speed,
    double acceleration,
    double pos,
    double size,
    double spawn_tick)
    : speed{speed},
      acceleration{acceleration},
      pos{pos},
      size{size},
      spawn_tick{spawn_tick},
      road(nullptr),
      node(nullptr),
      inner_timer{0},
      current_acceleration{0} {
    static int s_id{0};
    id = s_id;
    s_id++;
}

// void Movable::setRoad(Road* r)
void Movable::updatePosition() {
    this->pos = this->nxt_pos;
    this->speed = this->nextSpeed();
}
double Movable::nextPos() {
    double sp = this->speed + TIME * this->current_acceleration;
    double speedLimit = road->getSpeedLimit();
    if (sp > speedLimit && current_acceleration > 0) {
        double t = (speedLimit - speed) / current_acceleration;
        return MAX(t * TIME * this->current_acceleration / 2 + speed * t + speedLimit * (TIME - t) + this->pos, pos);
    } else if (sp < 0) {
        double t = -speed / current_acceleration;
        return MAX(t * TIME * this->current_acceleration / 2 + speed * t + this->pos, pos);
    } else {
        return MAX(TIME * TIME * current_acceleration / 2 + speed * TIME + pos, pos);
    }
}

double Movable::nextSpeed() {
    double sp = speed + TIME * current_acceleration;
    if (inner_timer > 0) {
        return 0;
    }
    double speedLimit = road->getSpeedLimit();
    if (sp > speedLimit && current_acceleration > 0) {
        return speedLimit;
    } else if (sp < 0) {
        return 0;
    } else {
        return sp;
    }
}
auto Movable::nextPosition() {
    double sp = this->speed + TIME * this->current_acceleration;
    double speedLimit = road->getSpeedLimit();
    struct result {
        double pos;
        double speed;
    };
    double futur_pos, futur_speed;
    if (inner_timer > 0) {
        return result{this->pos, this->speed};
    }
    if (sp > speedLimit && current_acceleration > 0) {
        double t = (speedLimit - speed) / current_acceleration;
        futur_pos = t * TIME * this->current_acceleration / 2 + speed * t + speedLimit * (TIME - t) + this->pos;
        futur_speed = speedLimit;
    } else if (sp < 0) {
        double t = -speed / current_acceleration;
        futur_pos = t * TIME * this->current_acceleration / 2 + speed * t + this->pos;
        futur_speed = 0;
    } else {
        futur_pos = TIME * TIME * current_acceleration / 2 + speed * TIME + pos;
        futur_speed = sp;
    };

    if (futur_pos < pos) {
        futur_pos = pos;
    }
    return result{futur_pos,
                  futur_speed};
}

// #define STOP_SPEED 2.5
void Movable::handleRoadTarget() {
    double dx = this->pos_goal - this->pos;
    if (this->road == this->road_goal && 0 < dx && dx < LEAVING_DIST) {
        double da = 0;
        if (speed < STOP_SPEED) {
            return;
        }
        da = (STOP_SPEED - speed - TIME * current_acceleration) / TIME;
        if (da > 0) {
            da = 0;
        }
        current_acceleration += da;
        nxt_pos = this->nextPos();
    }
}
void Movable::handleFirstMovable() {
    nxt_pos = nextPos();
    current_acceleration = acceleration;

    if (this->pos >= this->nxt_pos) {
        current_acceleration = acceleration;

        nxt_pos = nextPos();
    }

    double dx = this->road->roadLen - nxt_pos;
    if (current_acceleration < 0) {
        current_acceleration = 0;
        nxt_pos = nextPos();
    }

    if (road->blockTraffic) {
        double da = dx > 0 ? 1.75 * dx / TIME / TIME : 2.5 * dx / TIME / TIME;
        current_acceleration = MIN(acceleration, current_acceleration + da);
        nxt_pos = nextPos();
        return;
    }

    if (nxt_pos > road->roadLen && path.size() > 0) {
        if (!road->end->positionAvailable(road->posEnd, size)) {
            double da = 2.5 * dx / TIME / TIME;
            current_acceleration = MIN(acceleration, current_acceleration + da);
            nxt_pos = nextPos();
        }
    }
}
void Movable::handlePossibleCollision(Movable& other) {
    double da = 2.5 * ((other.nxt_pos - other.size) - (nxt_pos + size)) / TIME / TIME;
    if (da <= 0) {
        current_acceleration += da;
    } else {
        current_acceleration -= da;
    }
    nxt_pos = nextPos();
}
void Movable::noPossibleCollision(Movable& other) {
    if (current_acceleration < 0) {
        current_acceleration = 0;
        nxt_pos = nextPos();
    }

    double futureOther = other.nxt_pos - other.size;
    double futureSelf = nxt_pos - size;
    double dx = MIN(futureOther - futureSelf, futureOther - futureSelf - size);

    double da = 1.75*dx/TIME/TIME;
    double damax = (road->speedLimit - speed - TIME*current_acceleration)/TIME;
    da = MIN(da, damax);
    current_acceleration = MIN(acceleration, current_acceleration + da);
    nxt_pos = nextPos();
}
bool Movable::updateRoad() {
    // std::cout << "debug me : " << nxt_pos << " " << pos << std::endl;
    pos = nxt_pos;
    speed = nextSpeed();

    if (road == road_goal && pos > pos_goal) {
        double timer = inner_timer * TIME;
        if (timer <= LEAVING_TIME) {
            speed = 0;
            acceleration = 0;
            current_acceleration = 0;
            inner_timer++;
            return true;
        }
        road->despawnMovable(this);
        node = nullptr;
        return false;
    }

    if (pos >= road->roadLen && !road->blockTraffic) {
        Node* next_node = path.back();
        path.pop_back();
        node = next_node;
        node->movables.push_back(this);

        Road* next_road = findNextRoad(path.back());
        road->removeMovable(this);
        pos = 0;
        node_pos = road->posEnd;
        double norm = node_pos.getLength(next_road->posStart);
        node_dir = next_road->posStart.minus(road->posEnd);
        node_dir = node_dir.multiply(1 / norm);
        speed /= 2; // TODO modifié cela pour avoir un comportement réaliste
        current_acceleration = 1;
        node_len = norm;

        road = next_road;
    }
    return true;
}
void Movable::notifyNodeCollision() {
    speed = 0;
    current_acceleration = 0;
    node_mov = false;
}
void Movable::notifyNodePriority() {
    current_acceleration = acceleration / 2;
}
resNodePos Movable::nextNodePosition() {
    auto res = nextPosition();
    double pos = MIN(res.pos, node_len);

    Point p = node_dir.multiply(pos - this->pos);
    p = p.add(node_pos);
    return resNodePos{pos, res.speed, p};
}
void Movable::updateNode() {
    double tmpPos = pos;
    Point tmpNodePos = node_pos;

    auto res = nextNodePosition();
    // pos, speed, node_pos = res.pos, res.speed, res.p;
    pos = res.pos;
    speed = res.speed;
    node_pos = res.p;
    if (node_mov && pos <= tmpPos) {
        current_acceleration = acceleration;
    }
    if (pos >= node_len) {
        pos = 0;
        if (!road->addMovable(this)) {
            pos = node_len;
            node_pos = tmpNodePos;
        } else {
            // std::cout << "add mov is ok" << std::endl;
            node->movables.erase(std::remove(node->movables.begin(), node->movables.end(), this), node->movables.end());
            node = nullptr;
        }
    }
    node_mov = true;
}
bool Movable::update() {
    // std::cout << "mov :" << pos << std::endl;
    if (node == nullptr and road != nullptr) {
        return updateRoad();
    } else if (node != nullptr) {
        updateNode();
        return true;
    }
    return false;
}
void Movable::setRoad(Road* road) {
    this->road = road;
}
void Movable::stop() {
    speed = 0;
}

void Movable::setRoadPath(Road* arrival) {
    path = std::vector<Node*>();
    path.push_back(arrival->end);
    auto v2 = road->end->findPath(arrival->start);
    path.insert(path.end(), v2.begin(), v2.end());
}
void Movable::setRoadGoal(Road* arrival, double pos) {
    road_goal = arrival;
    pos_goal = pos;
    setRoadPath(arrival);
}
Road* Movable::findNextRoad(Node* nextNode) {
    auto currentNode = road->end;
    auto nextRoad = currentNode->roadTo(nextNode);
    return nextRoad;
}

double Movable::maxValue() {
    return pos + size;
}
double Movable::minValue() {
    return pos - size;
}
double Movable::getScore(double currentTick) {
    return ((currentTick - spawn_tick) * (currentTick - spawn_tick)) / 100;
}

double Movable::getPos() {
    return pos;
}
// int getId() const;

int Movable::getId() const {
    return id;
}

std::vector<double> Movable::toCoordXY() {
    if (node != nullptr) {
        // return node_pos;
        return std::vector<double>{node_pos.x, node_pos.y};
    }
    Point posStart = road->posStart;
    Point posEnd = road->posEnd;
    Point norm = posEnd.minus(posStart);
    norm = norm.multiply(pos / norm.getLength());
    Point p = norm.add(posStart);
    return std::vector<double>{p.x, p.y};
}