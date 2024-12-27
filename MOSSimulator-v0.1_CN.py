import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

class MOSSimulator:
    
    def info(self):

        """MOSSimulator类介绍"""

        print("\nMOSSimulator功能:\n ")
        print("1. 模拟MOS器件中表面电势和半导体空间电荷密度之间的关系。\n")
        print("2. 模拟MOS器件中绝缘层厚度、绝缘层介电常数和半导体掺杂浓度对C-V曲线的影响。\n")

    def __init__(self, T=300, tox=50e-9 * 100, eox=3.9, es=11.7, Na=1e16, Nd=1e15, Eg=1.12, Nc=2.8e19, Nv=2.65e19):

        # 用户自定义参数
        self.T = T
        self.tox = tox
        self.Nd = Nd
        self.Na = Na
        self.eox = eox
        self.es = es
        self.Eg = Eg
        self.Nc = Nc
        self.Nv = Nv

        # 物理常数
        self.e0 = 8.854e-14
        self.q = 1.602e-19
        self.k = 1.380649e-23
        
        # 材料参数
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
        二分法查找函数零点，即使得 f(x) = func(x) - y_target = 0 的x值。

        func: 所需函数。
        y_target: 所需函数对应的目标值.
        xmin: x的最小值.
        xmax: x的最大值.
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
        
        """计算热平衡费米能级"""
        
        Ev = 0
        Ec = self.Eg

        n = lambda Ef: self.Nc * np.exp(self.beta * (Ef - Ec))
        p = lambda Ef: self.Nv * np.exp(self.beta * (Ev - Ef))

        net_charge = lambda Ef: p(Ef) - n(Ef) + self.Nd - self.Na
        return self.solve_bisection(net_charge, 0, Ev, Ec)

    def set_tox(self, tox):
        
        """设置绝缘层厚度"""
        
        self.tox = tox

    def set_eox(self, eox):
        
        """设置绝缘层介电常数"""
        
        self.eox = eox

    def set_Na(self, Na):
        
        """设置半导体掺杂浓度并更新费米能级和平带电压"""
        
        self.Na = Na

        # 根据新的Na值更新费米能级
        self.Ef = self.compute_equilibrium_fermi_level()
        self.phi_s = self.chi_s + self.Eg - self.Ef
        self.phi_ms = self.phi_m - self.phi_s
        self.Vfb = self.phi_ms
    
    def ni(self):

        ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2 * self.k * self.T / self.q))  # 硅的本征载流子浓度 (cm^-3)
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

    def F(self, psis): # F函数

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
        绘制半导体空间电荷密度和表面势之间的关系图像。

        psis: 半导体表面势
        export: 选择是否导出数据
        """
        Qs_data = abs(self.space_charge_density(psis))
        
        if export: 
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
        根据Qs与Vs的关系计算出半导体Cd的电容进而计算出总电容
        return C / Ci versus total voltage V.
        """

        V = lambda psis: self.Vfb - self.space_charge_density(psis) / self.oxide_capacitance() + psis

        return V(psis), self.semiconductor_capacitance(psis) / (self.oxide_capacitance() + self.semiconductor_capacitance(psis))

    def save_data(self, filename, V, C):
        
        """将C-V关系保存为csv文件"""
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vg (V)', 'C / Ci'])
            for v, c in zip(V, C):
                writer.writerow([v, c])
        print(f"Data saved to {filename}.")

    def plot_Na_variation(self, psis, Na_values, export=False):
        
        """掺杂浓度对C-V曲线的影响"""
        
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
        
        """绝缘层介电常数对C-V曲线的影响"""
        
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
        
        """绝缘层厚度对C-V曲线的影响"""
        
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
        根据给定的 tox、eox、Na 和 T 计算 Cmin / Cox.
        """
        ln_term = np.log(self.Na / self.ni())
        
        # Calculate Cmin/Cox using the provided formula
        cmin_cox_ratio = 1 / (1 + (2 * self.eox) / (self.q * self.es * tox) * 
                             np.sqrt((self.es * self.e0 * self.k * self.T) / self.Na * ln_term))
        
        return cmin_cox_ratio

    def plot_cmin_cox_vs_tox(self, Na_values, tox_range=(10e-9, 10e-6), export=False):
        """
        绘制一系列不同 Na 值对应的 Cmin/Cox 与 tox 的关系图.
        
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
        根据给定的 tox、eox、Na 和 T 计算 CFB / Cox
        """
        term = np.sqrt((self.es * self.e0 * self.k * self.T) / self.Na )
        cfb_cox_ratio = 1 / (1 + (self.eox / (self.es * tox * self.q) * term))
        
        return cfb_cox_ratio

    def plot_cfb_cox_vs_tox(self, Na_values, tox_range=(10e-9, 10e-6), export=False):
        """
        绘制一系列不同 Na 值对应的 CFB/Cox 与 tox 的关系图。
        
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
        绘制一系列不同带隙材料的 x_dmax 与 Na 的关系图。
        
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

# 示例

psis = np.linspace(-0.4, 1.2, 500)

# 1. 绘制 Qs - psis 关系图
moss = MOSSimulator() # default parameters
moss.info()
moss.plot_charge_density(psis)

# 2. 绘制 C/Cox - V 关系图

# 2.1 改变 eox 
eox_values = [25.0, 20.0, 15.0, 10.0, 3.9]
moss.plot_eox_variation(psis, eox_values)

# 2.2.1 改变 Na(eox=3.9)
moss = MOSSimulator(tox=50e-7, eox=3.9)
Na_values = [5e17, 1e17, 5e16, 1e16]
moss.plot_Na_variation(psis, Na_values)

# 2.2.2 改变 Na(eox=25)
moss = MOSSimulator(tox=50e-7, eox=25)
Na_values = [5e17, 1e17, 5e16, 1e16]
moss.plot_Na_variation(psis, Na_values)

# 2.3.1 改变 tox(eox=3.9)
moss = MOSSimulator(Na=1e16, eox=3.9)
tox_values = [10e-7, 20e-7, 40e-7, 60e-7, 80e-7, 100e-7]
moss.plot_tox_variation(psis, tox_values)

# 2.3.2 改变 tox(eox=25)
moss = MOSSimulator(Na=1e16, eox=25)
tox_values = [10e-7, 20e-7, 40e-7, 60e-7, 80e-7, 100e-7]
moss.plot_tox_variation(psis, tox_values)


# 3.绘制 c/cox - tox 关系图
moss = MOSSimulator()

# 掺杂浓度示例
Na_values = [1e14, 3e14, 4e14, 5e14, 7e14, 1e15, 3e15, 4e15, 5e15, 7e15, 1e16, 3e16, 4e16, 5e16, 
             7e16, 1e17, 3e17, 4e17, 5e17, 7e17, 1e18, 3e18, 4e18, 5e18, 7e18]  
# 3.1 cmin/cox - tox
moss.plot_cmin_cox_vs_tox(Na_values) 

# 3.2 cfb/cox - tox
moss.plot_cfb_cox_vs_tox(Na_values)

# 4.绘制最大耗尽区宽度和掺杂浓度之间的关系

Eg_values = [0.66, 1.12, 1.42]  # 不同材料的带隙值示例
Nc_values = [1.04e19, 2.8e19, 4.7e17]
Nv_values = [6.1e19, 2.65e19, 7e18]
moss.plot_xdmax_vs_na(Eg_values, Nc_values, Nv_values)  