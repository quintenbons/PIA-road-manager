// main.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "movable.h"
#include <vector>

namespace py = pybind11;
double LEAVING_TIME = 5;
double LEAVING_DIST = 20;
double TIME = 0.5;
int MAX_MOVABLES_IN_NODE = 5;
double STOP_SPEED = 2.5;
double NODE_RADIUS = 10;
double ROAD_OFFSET = 2;

void setLeavingTime(double d) {
    LEAVING_TIME = d;
}

void setLeavingDist(double d) {
    LEAVING_DIST = d;
}

void setTime(double d) {
    TIME = d;
}

void setStopSpeed(double d) {
    STOP_SPEED = d;
}

void setMaxMovablesInNode(int i) {
    MAX_MOVABLES_IN_NODE = i;
}

void setNodeRadius(double d) {
    NODE_RADIUS = d;
}

void setRoadOffset(double d) {
    ROAD_OFFSET = d;
}

std::vector<int> spawnerUpdate(std::vector<Movable*> movables) {
    std::vector<int> removeList {};
    for (int index = 0; index < movables.size(); ++index) {
        Movable* mov = movables[index];
        if(!mov->update()) {
            removeList.push_back(index);
        }
    }

    return removeList;
}


PYBIND11_MODULE(engine_ia, m) {

    m.def("spawner_update", &spawnerUpdate);
    m.def("set_leaving_time", &setLeavingTime);
    m.def("set_leaving_dist", &setLeavingDist);
    m.def("set_time", &setTime);
    m.def("set_stop_speed", &setStopSpeed);
    m.def("set_max_movables_in_node", &setMaxMovablesInNode);
    m.def("set_node_radius", &setNodeRadius);
    m.def("set_road_offset", &setRoadOffset);

    py::class_<Movable>(m, "Movable")
        .def(py::init<double, double, double, double, double>())
        .def("get_id", &Movable::getId)
        .def("update", &Movable::update)
        .def("get_score", &Movable::getScore)
        .def("set_road_goal", &Movable::setRoadGoal)
        .def("get_pos", &Movable::getPos)
        .def("to_coord_xy", &Movable::toCoordXY);


    py::class_<Road>(m, "Road")
        .def(py::init<Node*, Node*, double>())
        .def("get_id", &Road::getId)
        .def("update", &Road::update)
        .def("get_pos_end", &Road::getPosEnd)
        .def("get_pos_start", &Road::getPosStart)
        .def("get_road_len", &Road::getRoadLen)
        .def("spawn_movable", &Road::spawnMovable)
        .def("get_block_traffic", &Road::getBlockTraffic)
        .def("set_block_traffic", &Road::setBlockTraffic)
        .def("get_ai_flow_count_0", &Road::getAiFlowCount0)
        .def("get_ai_flow_count_1", &Road::getAiFlowCount1)
        ;

    py::class_<Node>(m, "Node")
        .def(py::init<double, double>())
        .def("update", &Node::update)
        .def("get_id", &Node::getId)
        .def("get_x", &Node::getX)
        .def("get_y", &Node::getY)
        .def("set_position", &Node::setPosition)
        .def("set_path", &Node::constructPath)
        .def("get_road_in", &Node::getRoadIn)
        .def("get_road_out", &Node::getRoadOut)
        ;

        // .def("get_movables", &Road::getMovables, py::return_value_policy::reference_internal);
}
