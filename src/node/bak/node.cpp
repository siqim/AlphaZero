#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;


class Node {
public:
    Node(Node* parent, double p, int player_id, int num_actions) {
        this->parent = parent;
        this->actions.reserve(num_actions);
        this->child_nodes.reserve(num_actions);

        this->p = p;
        this->Q = 0.0;
        this->N = 0;
        this->num_actions = num_actions;

        this->player_id = player_id;
    }

public:
    double p, Q;
    int player_id, N, num_actions;
    Node* parent;
    std::vector<int> actions;
    std::vector<Node> child_nodes;

public:
    int get_next_player_id() {
        if (this->player_id == 1) {
            return 2;
        }
        else {
            return 1;
        }
    }

    void expand(std::vector<int> actions, std::vector<float> probs) {
        int next_player_id = get_next_player_id();
        size_t num_valid_actions = actions.size();

        for (int i = 0; i < num_valid_actions; i++) {
            this->actions.push_back(actions[i]);
            this->child_nodes.push_back(Node(this, probs[i], next_player_id, this->num_actions));
        }
    }

};


double calc_ucb(Node node, float c_puct) {
    return node.Q + c_puct * node.p * sqrt(node.parent->N) / (1.0 + node.N);
}

int get_max_ucb_child(Node node, float c_puct) {
    double best_U = -INFINITY;
    int best_idx = NULL;

    size_t num_children = node.child_nodes.size();
    for (int i = 0; i < num_children; i++) {
        double U = calc_ucb(node.child_nodes[i], c_puct);
        if (U > best_U) {
            best_U = U;
            best_idx = i;
        }
    }

    return best_idx;
}

void update_Q(Node& node, double v) {
    node.Q = (node.Q * node.N + v) / (1.0 + node.N);
}

void update_N(Node& node) {
    node.N++;
}

bool is_leaf_node(Node node) {
    if (node.actions.empty()) {
        return true;
    }
    else {
        return false;
    }
}

void clear_parent_info(Node& node) {
    delete node.parent;
    node.parent = NULL;
}

PYBIND11_MODULE(pynode, m) {
    py::class_<Node>(m, "Node")
        .def(py::init<Node*, double, int, int>())
        .def_readwrite("p", &Node::p)
        .def_readwrite("Q", &Node::Q)
        .def_readwrite("N", &Node::N)
        .def_readwrite("parent", &Node::parent)
        .def_readwrite("player_id", &Node::player_id)
        .def_readwrite("actions", &Node::actions)
        .def_readwrite("child_nodes", &Node::child_nodes)
        .def_readwrite("num_actions", &Node::num_actions)
        .def("expand", &Node::expand);

    m.def("calc_ucb", &calc_ucb);
    m.def("update_Q", &update_Q);
    m.def("update_N", &update_N);
    m.def("is_leaf_node", &is_leaf_node);
    m.def("clear_parent_info", &clear_parent_info);
    m.def("get_max_ucb_child", &get_max_ucb_child);

}
