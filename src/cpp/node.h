#ifndef NODE_H
#define NODE_H

#include "point.h"
#include <vector>
#include <map>
#include "road.h"
#include "movable.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;

class Road;
class Movable;

class Node {
   private:
    /* data */
   public:
    Node(double x, double y);

    void update(py::object obj, double timer);
    bool positionAvailable(Point& pos, double size);
    void addRoadIn(Road* road);
    void addRoadOut(Road* road);
    void setPath(std::unordered_map<Node*, Node*> p);
    void constructPath(Node* n1, Node* n2);
    int getId();
    std::vector<Road*> getRoadIn();
    std::vector<Road*> getRoadOut();
    double getX();
    double getY();

    void setPosition(double x, double y);

    std::vector<Node*> findPath(Node* other);
    Road* roadTo(Node* nextNode);

    Point position;
    std::vector<Road*> roadIn;
    std::vector<Road*> roadOut;

    std::unordered_map<Node*, Node*> paths;
    std::vector<Movable*> movables;
    int id;
};

#endif