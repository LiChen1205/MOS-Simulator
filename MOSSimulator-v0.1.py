import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

class MOSSimulator:
    
    def info(self):

        """Introduction to MOSSimulator."""

        print("\nFunction of MOSSimulator:\n ")
        print("1. Simulate the relationship between surface potential and space charge density of semiconductor in the metal-oxide-semiconductor system.\n")
        print("2. Simulate the impact of oxide thinkness, oxide dielectric constant and semiconductor doping concentration on the C-V curve.\n")

    def __init__(self, T=300, tox=50e-9 * 100, eox=3.9, es=11.7, Na=1e16, Nd=1e15, Eg=1.12, Nc=2.8e19, Nv=2.65e19):

        # User-defined interfaces
        self.T = T
        self.tox = tox
        self.Nd = Nd
        self.Na = Na
        self.eox = eox
        self.es = es
        self.Eg = Eg
        self.Nc = Nc
        self.Nv = Nv

        # Constants
        self.e0 = 8.854e-14
        self.q = 1.602e-19
        self.k = 1.380649e-23
        
        # Material parameters
        self.chi_s = 4.17
        self.phi_m = 5.01
        self.beta = self.q / (self.k * self.T)
        self.phit = 1 / (self.k * self.T)
        
        self.Ef = self.compute_equilibrium_fermi_level()
        self.phi_s = self.chi_s + self.Eg - self.Ef
        self.phi_ms = self.phi_m - self.phi_s
        self.Vfb = self.phi_ms

    def solve_bisection(self, func, y_target, xmin, xmax):

        """
        Use bisection method to search the zero point of f(x) = func(x) - y_target = 0.

        func: function needed.
        y_target: value of func(x) needed.
        xmin: Minimum value of x.
        xmax: Maximum value of x.
        """

        threshold_value = 1e-10
        max_iters = 100
        cnt = 0
        a, b = xmin, xmax
        fa, fb = func(a) - y_target, func(b) - y_target

        while np.abs(a - b) > threshold_value and cnt < max_iters:
            cnt += 1
            c = (a + b) / 2.0
            fc = func(c) - y_target
            if fc == 0:
                return c
            elif np.sign(fa) == np.sign(fc):
                a = c
                fa = fc
            else:
                b = c
        return (a + b) / 2.0

    def compute_equilibrium_fermi_level(self):
        
        """Compute the equilibrium fermi level."""
        
        Ev = 0
        Ec = self.Eg

        n = lambda Ef: self.Nc * np.exp(self.beta * (Ef - Ec))
        p = lambda Ef: self.Nv * np.exp(self.beta * (Ev - Ef))

        net_charge = lambda Ef: p(Ef) - n(Ef) + self.Nd - self.Na
        return self.solve_bisection(net_charge, 0, Ev, Ec)

    def set_tox(self, tox):
        
        """Set the thickness of oxide."""
        
        self.tox = tox

    def set_eox(self, eox):
        
        """Set the dielectric constant of oxide."""
        
        self.eox = eox

    def set_Na(self, Na):
        
        """Set Na and use Na to update the fermi level and flat band voltage."""
        
        self.Na = Na

        # Update Fermi level based on new Na value
        self.Ef = self.compute_equilibrium_fermi_level()
        self.phi_s = self.chi_s + self.Eg - self.Ef
        self.phi_ms = self.phi_m - self.phi_s
        self.Vfb = self.phi_ms
    
    def ni(self):

        ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2 * self.k * self.T / self.q))  # Intrinsic carrier concentration for silicon (cm^-3)
        return ni
    
    def psi_bp(self):

        psi_bp = (self.k * self.T * np.log(self.Na / self.ni())) / self.q
        return psi_bp
    
    def pp0(self):

        pp0 = self.Nv * np.exp(self.beta * (0 - self.Ef))
        return pp0
    
    def np0(self):

        np0 = self.Nc * np.exp(self.beta * (self.Ef - self.Eg))
        return np0

    def F(self, psis): # F function

        F = np.sqrt((np.exp(-self.beta * psis) + self.beta * psis - 1) + self.np0() / self.pp0() * (np.exp(self.beta * psis) - self.beta * psis - 1))

        return F
    
    def debye_length(self):

        Ld = np.sqrt(self.es * self.e0 / (self.q * self.pp0() * self.beta))
        return Ld
    
    def space_charge_density(self, psis):

        Qs = -np.sign(psis) * (np.sqrt(2) * self.es * self.e0 * self.k * self.T) / (self.q * self.debye_length()) * self.F(psis)
        return Qs

    def plot_charge_density(self, psis, export=False):
        """
        Plot the space charge density of semiconductor versus the surface potential.

        psis: surface potential of semiconductor.
        export: export the data or not.
        """
        Qs_data = abs(self.space_charge_density(psis))
        
        if export: # Save to Excel
            df = pd.DataFrame({
                '$V_s$ (V)': psis,
                '$Q_s$ (C / cm$^2$)': Qs_data
            })
            df.to_excel('charge_density_data.xlsx', index=False)

        plt.figure(dpi=600)
        plt.semilogy(psis, Qs_data)
        plt.axvline(x = self.psi_bp(), linestyle=':', label=f"$V_s$ = $V_{{BP}})$")
        plt.axvline(x = 2 * self.psi_bp(), linestyle='--', label=f"$V_s$ = $2V_{{BP}})$")
        plt.xlim(-0.4, 1.2)
        plt.title('Surface Space Charge Density ($Q_s$ vs $V_s$)')
        plt.xlabel('$V_s$ (V)')
        plt.ylabel('$Q_s$ (C / cm$^2$)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def oxide_capacitance(self):

        Cox = self.eox * self.e0 / self.tox
        return Cox
    
    def semiconductor_capacitance(self, psis):

        Cd = np.sign(psis) * self.es * self.e0 * (1 - np.exp(-self.beta * psis) + self.np0() / self.pp0() * (np.exp(self.beta * psis) - 1)) / (np.sqrt(2) * self.debye_length() * self.F(psis))
        return Cd

    def plot_capacitance(self, psis):
        """
        According to the relationship of Qs and Vs, calculate the capacitance of semiconductor Cd and then the total capacitance.
        return C / Ci versus total voltage V.
        """

        V = lambda psis: self.Vfb - self.space_charge_density(psis) / self.oxide_capacitance() + psis

        return V(psis), self.semiconductor_capacitance(psis) / (self.oxide_capacitance() + self.semiconductor_capacitance(psis))

    def save_data(self, filename, V, C):
        
        """Save V and C data to a CSV file."""
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vg (V)', 'C / Ci'])
            for v, c in zip(V, C):
                writer.writerow([v, c])
        print(f"Data saved to {filename}.")

    def plot_Na_variation(self, psis, Na_values, export=False):
        
        """Impact of doping concentration on C-V curve."""
        
        plt.figure(dpi=600)
        for Na in Na_values:
            self.set_Na(Na)
            V, C = self.plot_capacitance(psis)
            plt.plot(V, C, label=f'$N_A$={Na:.1e}')
            if export:
                self.save_data(f"capacitance_Na_{Na:.1e}.csv", V, C)
        plt.xlim(-2, 3)
        plt.ylim(0, 1.2)
        plt.title('Capacitance vs $V_{G}$ ($N_A$ variation)')
        plt.xlabel('$V_{G}$ (V)')
        plt.ylabel('C / $C_{i}$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_eox_variation(self, psis, eox_values, export=False):
        
        """Impact of dielectric constant on C-V curve."""
        
        plt.figure(dpi=600)
        for eox in eox_values:
            self.set_eox(eox)
            V, C = self.plot_capacitance(psis)
            plt.plot(V, C, label=f'$\epsilon_i$={eox}')
            if export:
                self.save_data(f"capacitance_eox_{eox}.csv", V, C)
        plt.xlim(-2, 3)
        plt.ylim(0, 1.2)
        plt.title('Capacitance vs $V_{G}$ ($\epsilon_i$ variation)')
        plt.xlabel('$V_{G}$ (V)')
        plt.ylabel('C / $C_{i}$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_tox_variation(self, psis, tox_values, export=False):
        
        """Impact of oxide thinkness on C-V curve."""
        
        plt.figure(dpi=600)
        for tox in tox_values:
            self.set_tox(tox)
            V, C = self.plot_capacitance(psis)
            plt.plot(V, C, label=f'$d_i$={tox:.1e}')
            if export:
                self.save_data(f"capacitance_tox_{tox:.1e}.csv", V, C)
        plt.xlim(-2, 3)
        plt.ylim(0, 1.2)
        plt.title('Capacitance vs $V_{G}$ ($d_{i}$ variation)')
        plt.xlabel('$V_{G}$ (V)')
        plt.ylabel('C / $C_{i}$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def cmin_cox(self, tox): # ratio of minimum capacitance in high frequency and oxide capacitance
        """
        Calculate the minimum capacitance ratio (Cmin / Cox) based on given tox, eox, Na, and T.
        """
        ln_term = np.log(self.Na / self.ni())
        
        # Calculate Cmin/Cox using the provided formula
        cmin_cox_ratio = 1 / (1 + (2 * self.eox) / (self.q * self.es * tox) * 
                             np.sqrt((self.es * self.e0 * self.k * self.T) / self.Na * ln_term))
        
        return cmin_cox_ratio

    def plot_cmin_cox_vs_tox(self, Na_values, tox_range=(10e-9, 10e-6), export=False):
        """
        Plot Cmin/Cox versus tox for a series of Na values.
        
        Na_values: List of doping concentrations.
        tox_range: Tuple containing the minimum and maximum values of tox (in meters).
        export: If True, save data to CSV files.
        """
        plt.figure(dpi=600)
        tox_values = np.linspace(tox_range[0], tox_range[1], 500)  # Generate tox values between 10nm and 10000nm
        
        for Na in Na_values:
            self.set_Na(Na)  # Set the current Na value
            cmin_cox_values = [self.cmin_cox(tox) for tox in tox_values]  # Calculate Cmin/Cox for each tox value
            
            # Plot the data
            plt.semilogy(tox_values * 1e9, cmin_cox_values, label=f'$N_A$={Na:.1e}')  # Convert tox to nm for plotting
            
            if export:
                # Save data to CSV
                filename = f'cmin_cox_Na_{Na:.1e}.csv'
                self.save_data(filename, tox_values * 1e9, cmin_cox_values)

        plt.xscale('log')
        plt.xlabel('Oxide Thickness $d_{i}$ (nm)')
        plt.ylabel('$C_{min}/C_{i}$')
        plt.title('$C_{min}/C_{i}$ vs $d_{i}$ for different $N_A$ values')
        plt.legend()
        plt.grid(True)
        plt.xlim(100, 1e4)
        plt.ylim(0.01, 1)
        plt.show()
    
    def cfb_cox(self, tox): # ratio of total flatband capacitance and oxide capacitance
        """
        Calculate the flat-band capacitance ratio (CFB / Cox) based on given tox, eox, Na, and T.
        """
        term = np.sqrt((self.es * self.e0 * self.k * self.T) / self.Na )
        cfb_cox_ratio = 1 / (1 + (self.eox / (self.es * tox * self.q) * term))
        
        return cfb_cox_ratio

    def plot_cfb_cox_vs_tox(self, Na_values, tox_range=(10e-9, 10e-6), export=False):
        """
        Plot CFB/Cox versus tox for a series of Na values.
        
        Na_values: List of doping concentrations.
        tox_range: Tuple containing the minimum and maximum values of tox (in meters).
        export: If True, save data to CSV files.
        """
        plt.figure(dpi=600)
        tox_values = np.linspace(tox_range[0], tox_range[1], 500)  # Generate tox values between 10nm and 10000nm
        
        for Na in Na_values:
            self.set_Na(Na)  # Set the current Na value
            cfb_cox_values = [self.cfb_cox(tox) for tox in tox_values]  # Calculate CFB/Cox for each tox value
            
            # Plot the data
            plt.plot(tox_values * 1e9, cfb_cox_values, label=f'$N_A$={Na:.1e}')  # Convert tox to nm for plotting
            
            if export:
                # Save data to CSV
                filename = f'cfb_cox_Na_{Na:.1e}.csv'
                self.save_data(filename, tox_values * 1e9, cfb_cox_values)

        plt.xscale('log')
        plt.xlabel('Oxide Thickness $d_{i}$ (nm)')
        plt.ylabel('$C_{FB}/C_{i}$')
        plt.title('$C_{FB}/C_{i}$ vs $d_{i}$ for different $N_A$ values')
        plt.legend()
        plt.grid(True)
        plt.xlim(100, 1e4)
        plt.ylim(0.3, 1.0)
        plt.show()

    def plot_xdmax_vs_na(self, Eg_values, Nc_values, Nv_values, Na_range=(1e14, 1e18), export=False):
        """
        Plot x_dmax versus Na for a series of different bandgap materials.
        
        Eg_values: List of bandgap values (in eV) for different materials.
        Nc_values: List of conduction band effective density of states for different materials.
        Nv_values: List of valence band effective density of states for different materials.
        Na_range: Tuple containing the minimum and maximum Na values.
        export: If True, save data to CSV files.
        """
        plt.figure(dpi=600)
        Na_values = np.logspace(np.log10(Na_range[0]), np.log10(Na_range[1]), 500)  # Generate Na values between 1e14 and 1e18

        for Eg, Nc, Nv in zip(Eg_values, Nc_values, Nv_values):
            # Update intrinsic carrier concentration (ni) based on given Nc and Nv for each material
            ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * self.k * self.T / self.q))

            # Define xdmax function for the current material
            def xdmax(Na):
                return np.sqrt((4 * self.es * self.e0 * self.k * self.T) / (self.q**2 * Na) * np.log(Na / ni))

            # Calculate xdmax for each Na value in the range
            xdmax_values = [xdmax(Na) for Na in Na_values]

            # Plot xdmax vs Na for this specific material
            plt.plot(Na_values, xdmax_values, label=f'$E_g$={Eg:.2f} eV')

            if export:
                # Save data to CSV
                filename = f'xdmax_Eg_{Eg:.2f}.csv'
                self.save_data(filename, Na_values, xdmax_values)

        # Configure plot settings
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Doping Concentration $N_A$ (cm$^{-3}$)')
        plt.ylabel('$x_{dmax}$ (cm)')
        plt.title('$x_{dmax}$ vs $N_A$ for Different Bandgap Materials')
        plt.legend()
        plt.grid(True)
        plt.xlim(1e14, 1e18)
        plt.ylim(1e-6, 1e-3)
        plt.show()

# Example usage

psis = np.linspace(-0.4, 1.2, 500)

# 1. Plot Qs vs psis
moss = MOSSimulator() # default parameters
moss.info()
moss.plot_charge_density(psis)

# 2. Plot C/Cox vs V

# 2.1 eox variation
eox_values = [25.0, 20.0, 15.0, 10.0, 3.9]
moss.plot_eox_variation(psis, eox_values)

# 2.2.1 Na variation(eox=3.9)
moss = MOSSimulator(tox=50e-7, eox=3.9)
Na_values = [5e17, 1e17, 5e16, 1e16]
moss.plot_Na_variation(psis, Na_values)

# 2.2.2 Na variation(eox=25)
moss = MOSSimulator(tox=50e-7, eox=25)
Na_values = [5e17, 1e17, 5e16, 1e16]
moss.plot_Na_variation(psis, Na_values)

# 2.3.1 tox variation(eox=3.9)
moss = MOSSimulator(Na=1e16, eox=3.9)
tox_values = [10e-7, 20e-7, 40e-7, 60e-7, 80e-7, 100e-7]
moss.plot_tox_variation(psis, tox_values)

#2.3.2 tox variation(eox=25)
moss = MOSSimulator(Na=1e16, eox=25)
tox_values = [10e-7, 20e-7, 40e-7, 60e-7, 80e-7, 100e-7]
moss.plot_tox_variation(psis, tox_values)


# 3.Plot c/cox vs tox
moss = MOSSimulator()

# Example doping concentrations
Na_values = [1e14, 3e14, 4e14, 5e14, 7e14, 1e15, 3e15, 4e15, 5e15, 7e15, 1e16, 3e16, 4e16, 5e16, 
             7e16, 1e17, 3e17, 4e17, 5e17, 7e17, 1e18, 3e18, 4e18, 5e18, 7e18]  
# 3.1 cmin/cox vs tox
moss.plot_cmin_cox_vs_tox(Na_values)  # tox from 10nm to 10,000nm 

# 3.2 cfb/cox vs tox
moss.plot_cfb_cox_vs_tox(Na_values)

# 4.Plot maximum depletion region width vs doping concentration 

Eg_values = [0.66, 1.12, 1.42]  # Example bandgap values for different materials
Nc_values = [1.04e19, 2.8e19, 4.7e17]
Nv_values = [6.1e19, 2.65e19, 7e18]
moss.plot_xdmax_vs_na(Eg_values, Nc_values, Nv_values)  # Na from 1e14 to 1e18 cm^-3