#include "Python.h"
// ��������
float ucb(float Q, float c_puct, float p, int N_parent, int N) {
    return Q + c_puct * p * sqrt(N_parent) / (1 + N);
}

// ��������
static PyObject* Exten_ucb(PyObject* self, PyObject* args) {
    float Q, c_puct, p;
    int N_parent, N;
    // ��ȡ���ݣ�i����int��ii��������int
    // ���û�л�ȡ�����򷵻�NULL
    if (!PyArg_ParseTuple(args, "fffii", &Q, &c_puct, &p, &N_parent, &N)) {
        return NULL;
    }

    return (PyObject*)Py_BuildValue("f", ucb(Q, c_puct, p, N_parent, N));
}

// ���PyMethodDef ModuleMethods[]����
static PyMethodDef ExtenMethods[] = {
    // add��������Python���õĺ�������Exten_add��C++�ж�Ӧ�ĺ�����
    {"ucb", Exten_ucb, METH_VARARGS},
    {NULL,NULL},
};

// ��ʼ������
static struct PyModuleDef ExtenModule = {
    PyModuleDef_HEAD_INIT,
    "UCB",//ģ������
    NULL,
    -1,
    ExtenMethods
};

void PyInit_UCB() {
    PyModule_Create(&ExtenModule);
}
