#include <pybind11/pybind11.h>
#include <pybind11/stl.h >
#include <vector>
#include <algorithm>
using std::vector;
using std::max;
using std::min;
namespace py = pybind11;


double calc_ucb(double c_puct, double parent_N, double Q, double p, double N) {
    return Q + c_puct * p * sqrt(parent_N) / (1.0 + N);
}

int get_max_ucb_child(double c_puct, int parent_N, vector<vector<double>> child_stats) {
    double best_U = -INFINITY;
    int best_idx = NULL;

    size_t num_children = child_stats.size();
    for (int i = 0; i < num_children; i++) {
        vector<double> child_stat = child_stats[i];
        double U = calc_ucb(c_puct, parent_N, child_stat[0], child_stat[1], child_stat[2]);
        if (U > best_U) {
            best_U = U;
            best_idx = i;
        }
    }

    return best_idx;
}

bool is_five_in_a_row(int x, int y, vector<vector<int>> state, int board_size, int player_id) {
    int counter = 0;
    for (int i = max(0, x - 5 + 1); i < min(board_size, x + 5); i++) {
        if (state[i][y] == player_id) {
            counter++;
        }
        else
        {
            counter = 0;
        }
        if (counter == 5) {
            return true;
        }
    }


    counter = 0;
    for (int i = max(0, y - 5 + 1); i < min(board_size, y + 5); i++) {
        if (state[x][i] == player_id) {
            counter++;
        }
        else
        {
            counter = 0;
        }
        if (counter == 5) {
            return true;
        }
    }



    counter = 0;
    for (int i = max({ -x, -y, 1 - 5 }); i < min({ board_size - x, board_size - y, 5 }); i++) {
        if (state[x + i][y + i] == player_id) {
            counter++;
        }
        else
        {
            counter = 0;
        }
        if (counter == 5) {
            return true;
        }
    }


    counter = 0;
    for (int i = max({ -x, y - board_size + 1, 1 - 5 }); i < min({ board_size - x, y + 1, 5 }); i++) {
        if (state[x + i][y - i] == player_id) {
            counter++;
        }
        else
        {
            counter = 0;
        }
        if (counter == 5) {
            return true;
        }
    }

    return false;

}


//
//void update_Q(Node& node, double v) {
//    node.Q = (node.Q * node.N + v) / (1.0 + node.N);
//}
//
//void update_N(Node& node) {
//    node.N++;
//}
//
//bool is_leaf_node(Node node) {
//    if (node.actions.empty()) {
//        return true;
//    }
//    else {
//        return false;
//    }
//}
//
//void clear_parent_info(Node& node) {
//    delete node.parent;
//    node.parent = NULL;
//}

PYBIND11_MODULE(pynode, m) {
    m.def("calc_ucb", &calc_ucb);
    m.def("get_max_ucb_child", &get_max_ucb_child);
    m.def("is_five_in_a_row", &is_five_in_a_row);
}
