#ifndef POINT_H
#define POINT_H

class Point
{
private:
public:
    double x, y;
    Point();
    Point(double x, double y);
    void setPos(double x, double y);
    auto getPos();
    double getLength(Point& p);
    double getLength();
    double scalaire(Point& p);
    Point ortho();
    Point minus(Point& p);
    Point add(Point& p);
    Point multiply(double lambda);
    Point normalize();

};

#endif
