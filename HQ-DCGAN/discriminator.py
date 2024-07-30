import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from math import ceil


n_qubits = 4  # Total number of qubits / N
n_layers = 1
kernel_size = n_qubits
stride = 4

dev = qml.device("default.qubit", wires=n_qubits)

def circuit(inputs, weights):
    var_per_qubit = int(len(inputs) / n_qubits) + 1
    encoding_gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        for i in range(var_per_qubit):
            if (qub * var_per_qubit + i) < len(inputs):
                exec('qml.{}({}, wires = {})'.format(encoding_gates[i], inputs[qub * var_per_qubit + i], qub))
            else:  # load nothing
                pass

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
            # qml.CNOT(wires = [i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[l, j], wires=j % n_qubits)

    _expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return _expectations

class Quanv1d(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super(Quanv1d, self).__init__()
        weight_shapes = {"weights": (n_layers, 2 * n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        assert len(X.shape) == 2
        bs = X.shape[0]
        XL = []
        for i in range(0, X.shape[1] - 4, stride):
                XL.append(self.ql1(torch.flatten(X[:, i:i + kernel_size], start_dim=1)))
        X = torch.cat(XL, dim=1).view(-1, 4, 16)
        return X


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.q1 = Quanv1d(kernel_size=kernel_size, stride=stride)
        self.c1 = nn.Conv1d(4, 64, 3, stride=1)
        self.c2 = nn.Conv1d(64, 128, 3, stride=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(64*14*60, 1)

    def forward(self, x):

        x = self.q1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.dropout1d(x)
        x = self.c1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.c2(x)
        x = self.bn3(x)
        x = x.view(-1, 64*14*60)
        x = self.fc(x)
        result = torch.sigmoid(x)
        return result