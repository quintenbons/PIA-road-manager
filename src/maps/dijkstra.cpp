// Small programme to calculate paths
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <list>

#include <limits>
#include <unordered_map>

class Road;
constexpr double infinity = std::numeric_limits<double>::infinity();

class Node {
    private:
    /* data */
    public:
        std::vector<Road*> roadIn;
        std::vector<Road*> roadOut;
        int id;
        Node(int id);
        // ~dijkstra();
        void addRoadIn(Road*);
        void addRoadOut(Road*);
};

class Road {
    public:
        Node *start;
        Node *end;
        double distance;

        Road(Node* start, Node* end, double distance);

};

Road::Road(Node* start, Node* end, double distance)
    : start {start}, end {end}, distance {distance} {
        start->addRoadOut(this);
        end->addRoadIn(this);
    }


Node::Node(int id)
    : id{id} {}

void Node::addRoadIn(Road* road) {
    this->roadIn.push_back(road);
}

void Node::addRoadOut(Road* road) {
    this->roadOut.push_back(road);
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

auto findMinDist(std::list<int>& Q, std::vector<double> d) {
    double m = infinity;
    auto minElm = Q.end();

    for(auto it = Q.begin(); it != Q.end(); ++it) {
        int q = *it;
        if(d[q] < m) {
            m = d[q];
            minElm = it;
        }
    }
    return minElm;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "./dijkstra name_of_file" << std::endl;
        return 0;
    }
    std::string fileName{argv[1]};
    std::ifstream file;

    file.open(fileName, std::ios::in);

    std::string line;
    std::getline(file, line);
    int nodeCount = std::stoi(line);

    std::vector<Node*> nodes;
    for(int index = 0; index < nodeCount; ++index) {
        nodes.push_back(new Node(index));
    }

    while (std::getline(file, line)) {
        std::vector<std::string> splited = split(line, ' ');

        int from = std::stoi(splited[0]);
        int to = std::stoi(splited[1]);
        double distance = std::stod(splited[2]);
        Road *r = new Road(nodes[from], nodes[to], distance);
        Road *r2 = new Road(nodes[to], nodes[from], distance);

    }

    for (Node* n : nodes) {
        std::vector<bool> usedNodes(nodeCount, false);
        std::vector<double> distances(nodeCount, infinity);
        std::unordered_map<int, int> previous;
        std::list<int> unusedNodes;
        for(Node* n : nodes) {
            unusedNodes.push_back(n->id);
        }
        distances[n->id] = 0.0;

        while(unusedNodes.size() > 0) {
            auto nodeRef = findMinDist(unusedNodes, distances);
            if(nodeRef == unusedNodes.end()) {
                std::cout << "Erreur de format" << std::endl;
                return 0;
            }
            int n = *nodeRef;

            unusedNodes.erase(nodeRef);
            usedNodes[n] = true;

            for(Road* r : nodes[n]->roadOut) {
                Node* other = r->end;
                double roadLen = r->distance;

                if(!usedNodes[other->id] && distances[other->id] > distances[n] + roadLen) {
                    distances[other->id] = distances[n] + roadLen;
                    if(previous.contains(other->id)) {
                        previous[other->id] = n;
                    }
                    else {
                        previous.insert({other->id, n});
                    }
                }

            }
        }

        std::cout << n->id << std::endl;
        for(auto i : previous) {
            std::cout << i.first << ":" << i.second << "|";
        }
        std::cout << std::endl;

    }

    for(Node *n : nodes) {
        for(Road *r : n->roadOut) {
            delete r;
        }
        delete n;
    }
    file.close();
    return 0;
}
