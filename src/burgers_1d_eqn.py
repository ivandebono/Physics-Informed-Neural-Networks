import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    """
    Physics-informed neural network model.
    Constructs a fully connected feedforward neural network with tanh
    activations between hidden layers. The default architecture maps a
    2-dimensional input (x, t) to a single scalar output u.
    Parameters:
        layers (list[int], optional): layer sizes including input and
            output dimensions. Default [2, 20, 20, 20, 20, 1].
    Forward input:
        x (Tensor): input tensor of shape (..., 2).
    Returns:
        Tensor: output tensor of shape (..., 1), representing the predicted u.
    """
    def __init__(self, layers=None):  # input: (x,t), output: u
        super().__init__()
        if layers is None:
            layers = [2, 20, 20, 20, 20, 1]
        self.layers = nn.ModuleList()
        for i in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Generate collocation points
N = 10000
x = torch.rand(N, 1) * 2*np.pi - np.pi  # domain [-pi, pi]
t = torch.rand(N, 1) * 1                # time [0,1]
xt = torch.cat([x, t], dim=1)
xt.requires_grad_()

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nu = 0.01 / np.pi  # viscosity

for epoch in range(100):
    if epoch % 10 == 0:
        print(f'Epoch {epoch}')
    u = model(xt)
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    # PDE residual
    residual = u_t + u * u_x - nu * u_xx

    # Only PDE residual is used in the loss; boundary/data terms are not implemented.
    # As a result, the model may not satisfy initial or boundary conditions, which can reduce solution accuracy.
    loss = torch.mean(residual**2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()