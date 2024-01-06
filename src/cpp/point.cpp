#include "point.h"

#include <cmath>
Point::Point() : x{0.0}, y{0.0} {}

Point::Point(double x, double y) : x{x}, y{y} {}

auto Point::getPos() {
    struct result {
        double x;
        double y;
    };
    return result{x, y};
}

void Point::setPos(double x, double y) {
    this->x = x;
    this->y = y;
}

double Point::getLength() {
    return sqrt((x * x) + (y * y));
}

double Point::getLength(Point& p) {
    double vx = (p.x - x);
    double vy = (p.y - y);
    return sqrt((vx * vx) + (vy * vy));
}

double Point::scalaire(Point& p) {
    return x * p.x + y * p.y;
}
Point Point::ortho() {
    return Point(y, -x);
}
Point Point::minus(Point& p) {
    return Point(x - p.x, y - p.y);
}
Point Point::add(Point& p) {
    return Point(x + p.x, y + p.y);
}
Point Point::multiply(double lambda) {
    return Point(x*lambda, y*lambda);
}