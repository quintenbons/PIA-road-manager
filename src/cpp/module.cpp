// main.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "movable.h"
#include <vector>
// #include "pet.h"

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
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

double next_pos(double speedLimit, double current_acceleration, double speed, double TIME, double pos) {
    double tca = TIME * current_acceleration;
    double sp = speed + tca;

    if (sp > speedLimit && current_acceleration > 0) {
        double t = (speedLimit - speed) / current_acceleration;
        return t * tca / 2 + speed * t + speedLimit * (TIME - t) + pos;
    } else if (sp < 0) {
        double t = -speed / current_acceleration;
        return t * tca / 2 + speed * t + pos;
    }
    return TIME*tca/2 + speed*TIME + pos;
}

PYBIND11_MODULE(engine_ia, m) {
    m.def("add", &add, "Add two numbers");
    m.def("next_pos", &next_pos, "next_position for movables");

    m.def("spawner_update", &spawnerUpdate);

    py::class_<Movable>(m, "Movable")
        .def(py::init<double, double, double, double, double, double>())
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
        .def("get_speed_limit", &Road::getSpeedLimit)
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
