import matplotlib.pyplot as plt  # Ensure matplotlib is installed and imported correctly

# ---------------------------
# Diagnostics helper
# ---------------------------
# 


class DiagnosticsLogger:
    def __init__(self):
        self.history = {
            "epoch": [],
            "L_total": [],
            "L_phys": [],
            "L_data": [],
            "L_ic": [],
            "zeta": [],
            "omega_n": []
        }
    def log(self, epoch, total_loss, physics_loss, data_loss, ic_loss, zeta, omega):
        self.history["epoch"].append(epoch)
        self.history["L_total"].append(total_loss)
        self.history["L_phys"].append(physics_loss)
        self.history["L_data"].append(data_loss)
        self.history["L_ic"].append(ic_loss)
        self.history["zeta"].append(zeta)
        self.history["omega_n"].append(omega)

    def plot(self):
        h = self.history

        # Losses (log scale)
        plt.figure(figsize=(11,5))
        plt.plot(h["epoch"], h["L_total"], label="Total loss")
        plt.plot(h["epoch"], h["L_phys"], label="Physics loss")
        plt.plot(h["epoch"], h["L_data"], label="Data loss")
        plt.plot(h["epoch"], h["L_ic"], label="IC loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN Loss Convergence")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Parameter trajectories
        plt.figure(figsize=(11,5))
        plt.plot(h["epoch"], h["zeta"], label="ζ(t)")
        plt.plot(h["epoch"], h["omega_n"], label="ωₙ(t)")
        plt.xlabel("Epoch")
        plt.ylabel("Parameter value")
        plt.title("Parameter Trajectories During Training")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()