#include "Python.h"
// 函数主体
float ucb(float Q, float c_puct, float p, int N_parent, int N) {
    return Q + c_puct * p * sqrt(N_parent) / (1 + N);
}

// 包裹函数
static PyObject* Exten_ucb(PyObject* self, PyObject* args) {
    float Q, c_puct, p;
    int N_parent, N;
    // 获取数据，i代表int，ii代表两个int
    // 如果没有获取到，则返回NULL
    if (!PyArg_ParseTuple(args, "fffii", &Q, &c_puct, &p, &N_parent, &N)) {
        return NULL;
    }

    return (PyObject*)Py_BuildValue("f", ucb(Q, c_puct, p, N_parent, N));
}

// 添加PyMethodDef ModuleMethods[]数组
static PyMethodDef ExtenMethods[] = {
    // add：可用于Python调用的函数名，Exten_add：C++中对应的函数名
    {"ucb", Exten_ucb, METH_VARARGS},
    {NULL,NULL},
};

// 初始化函数
static struct PyModuleDef ExtenModule = {
    PyModuleDef_HEAD_INIT,
    "UCB",//模块名称
    NULL,
    -1,
    ExtenMethods
};

void PyInit_UCB() {
    PyModule_Create(&ExtenModule);
}
