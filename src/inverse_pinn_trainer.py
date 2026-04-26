
class InversePINNTrainer:
    def __init__(self, n_epochs=2000, colloc_points=8000, seed=42, 
                 use_lbfgs=True, zeta_init=0.2, omega_n_init=3.0):
        self.seed = seed
        self.use_lbfgs = use_lbfgs
        self.zeta_init = zeta_init
        self.omega_n_init = omega_n_init
        self.n_epochs = n_epochs
        self.colloc_points = colloc_points
        self.model = None
        self.device = None
        self.t_raw = None
        self.t_measured = None
        self.dt_scale = None
        self.t_min = None
        self.t_max = None
        self.x_measured = None
        self.diag = None

    def to_t_hat(self, t_phys, t_min, t_max):
        """Normalize time to t_hat in [-1, 1]"""
        return 2.0 * (t_phys - t_min) / (t_max - t_min) - 1.0

    def train(self, observed_data, seed=None, n_epochs=None, colloc_points=None):
        """Train an inverse PINN to identify parameters of a damped harmonic oscillator."""
        if n_epochs is None:
            n_epochs = self.n_epochs
        if colloc_points is None:
            colloc_points = self.colloc_points
        if seed is None:
            seed = self.seed

        # Set reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t_data = get_time_observed(observed_data)
        x_data = get_x_observed(observed_data)

        self.t_raw = torch.tensor(np.asarray(t_data).reshape(-1, 1), 
                                  dtype=torch.float64, device=self.device)
        self.x_measured = torch.tensor(np.asarray(x_data).reshape(-1, 1), 
                                       dtype=torch.float64, device=self.device)

        self.t_min = float(self.t_raw.min().item())
        self.t_max = float(self.t_raw.max().item())
        if self.t_max == self.t_min:
            raise ValueError("t_data has zero range")

        self.t_measured = self.to_t_hat(self.t_raw, self.t_min, self.t_max).detach()
        self.dt_scale = 2.0 / (self.t_max - self.t_min)

        self.model = InversePINN(zeta_init=self.zeta_init, 
                                 omega_n_init=self.omega_n_init).to(self.device)
        self.colloc_points = colloc_points
        self.n_epochs = n_epochs

        self._run_training()
        return self.model, self.diag

    def _run_training(self):
        """Run the actual training loop."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.diag = DiagnosticsLogger()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            loss, Lp, Ld, Li = self.total_loss()
            loss.backward()
            optimizer.step()

            self.diag.log(epoch, loss.item(), Lp.item(), Ld.item(), Li.item(),
                             self.model.zeta.item(), self.model.omega_n.item())
            
            if epoch % 10 == 0:
                print(f"Epoch: {int(epoch)}, Total Loss: {loss:.2f}, Physics Loss: {Lp:.2f}, Data Loss: {Ld:.2f}, IC Loss: {Li:.2f}, Zeta: {self.model.zeta.item():.5f}, Omega: {self.model.omega_n.item():.5f}")

                

    def get_collocation(self):
        """Generate random collocation points for physics-informed training."""
        t = torch.rand(self.colloc_points, 1, dtype=torch.float64, device=self.device) * 2.0 - 1.0
        t.requires_grad_(True)
        return t

    def pde_residual(self, t_hat):
        """Compute the residual of the damped harmonic oscillator PDE."""
        t_hat = t_hat.clone().requires_grad_(True)
        x = self.model(t_hat)
        dx_dthat = torch.autograd.grad(x, t_hat, torch.ones_like(x), create_graph=True)[0]
        d2x_dthat2 = torch.autograd.grad(dx_dthat, t_hat, torch.ones_like(dx_dthat), create_graph=True)[0]

        x_t_phys = self.dt_scale * dx_dthat
        x_tt_phys = (self.dt_scale**2) * d2x_dthat2

        return x_tt_phys + 2.0 * self.model.zeta * self.model.omega_n * x_t_phys + (self.model.omega_n**2) * x

    def physics_loss(self):
        """Calculate physics loss based on PDE residuals."""
        t_colloc = self.get_collocation()
        r = self.pde_residual(t_colloc)
        return torch.mean(r**2)

    def data_loss(self):
        """Calculate MSE loss between model predictions and measured data."""
        pred = self.model(self.t_measured)
        return torch.mean((pred - self.x_measured)**2)

    def ic_loss(self):
        """Compute initial condition loss for displacement and velocity."""
        t0_phys = self.t_raw[0:1]
        t1_phys = self.t_raw[1:2] if self.t_raw.shape[0] >= 2 else self.t_raw[0:1] + 1e-6
        x0_meas = self.x_measured[0:1]
        x1_meas = self.x_measured[1:2] if self.x_measured.shape[0] >= 2 else self.x_measured[0:1]

        dt_phys = float((t1_phys - t0_phys).cpu().numpy().flatten()[0])
        v0_true = torch.zeros_like(x0_meas, device=self.device) if dt_phys == 0 else (x1_meas - x0_meas) / dt_phys

        t0_hat = self.to_t_hat(t0_phys, self.t_min, self.t_max).clone().requires_grad_(True).to(self.device)
        x0_pred = self.model(t0_hat)
        dx_dthat = torch.autograd.grad(x0_pred, t0_hat, torch.ones_like(x0_pred), create_graph=True)[0]
        x_t0_pred_phys = self.dt_scale * dx_dthat

        return torch.mean((x0_pred - x0_meas)**2) + torch.mean((x_t0_pred_phys - v0_true)**2)

    def total_loss(self):
        """Calculate total weighted loss."""
        Lp = self.physics_loss()
        Ld = self.data_loss()
        Li = self.ic_loss()
        w_d = 10.0
        w_i = 10.0
        return Lp + w_d * Ld + w_i * Li, Lp, Ld, Li

    def plot_phys_vs_data(self, observed_data, analytical_data=False):
        """Plot fit vs data on physical time."""
        t_plot_phys = np.linspace(self.t_min, self.t_max, 600).reshape(-1, 1)
        t_plot_phys_tensor = torch.tensor(t_plot_phys, dtype=torch.float64, device=self.device)
        t_plot_hat = self.to_t_hat(t_plot_phys_tensor, self.t_min, self.t_max).reshape(-1, 1)
        
        with torch.no_grad():
            x_pred = self.model(t_plot_hat).cpu().numpy().flatten()

        plt.figure(figsize=(11, 6))
        t_raw = get_time_observed(observed_data)
        x_measured = get_x_observed(observed_data)
        plt.plot(t_raw, x_measured, 'ro', label='Measured data', alpha=0.8, markersize=5)
        plt.plot(t_plot_phys.flatten(), x_pred, 'b-', linewidth=2.0, label='PINN prediction')
        
        if analytical_data:
            x_analytical = analytical(t_plot_phys, zeta=0.3, omega_n=5.0)
            plt.plot(t_plot_phys.flatten(), x_analytical, 'black', linewidth=2.0, label='Analytical solution')
        
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement")
        plt.title(f"PINN fit — ζ = {self.model.zeta.item():.5f} | ωₙ = {self.model.omega_n.item():.5f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
