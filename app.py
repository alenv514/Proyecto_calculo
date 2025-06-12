from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy import symbols, lambdify, sqrt, diff
import base64
import io
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class SimpsonIntegralCalculator:
    def __init__(self):
        self.x, self.y, self.z = symbols('x y z')
    
    def simpson_1d(self, func, a, b, n=1000):
        """Integración 1D usando regla de Simpson compuesta"""
        if n % 2 != 0:
            n += 1
        
        h = (b - a) / n
        x_vals = np.linspace(a, b, n + 1)
        y_vals = [func(x) for x in x_vals]
        
        integral = y_vals[0] + y_vals[n]
        
        for i in range(1, n):
            if i % 2 == 0:
                integral += 2 * y_vals[i]
            else:
                integral += 4 * y_vals[i]
        
        return integral * h / 3
    
    def simpson_2d(self, func, x_limits, y_limits, nx=50, ny=50):
        """Integración 2D usando regla de Simpson compuesta"""
        if nx % 2 != 0:
            nx += 1
        if ny % 2 != 0:
            ny += 1
        
        a, b = x_limits
        c, d = y_limits
        
        hx = (b - a) / nx
        hy = (d - c) / ny
        
        x_vals = np.linspace(a, b, nx + 1)
        y_vals = np.linspace(c, d, ny + 1)
        
        integral = 0
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                weight = 1
                
                if i == 0 or i == nx:
                    weight *= 1
                elif i % 2 == 1:
                    weight *= 4
                else:
                    weight *= 2
                
                if j == 0 or j == ny:
                    weight *= 1
                elif j % 2 == 1:
                    weight *= 4
                else:
                    weight *= 2
                
                integral += weight * func(x_vals[i], y_vals[j])
        
        return integral * hx * hy / 9
    
    def simpson_3d(self, func, x_limits, y_limits, z_limits, nx=20, ny=20, nz=20):
        """Integración 3D usando regla de Simpson compuesta"""
        if nx % 2 != 0:
            nx += 1
        if ny % 2 != 0:
            ny += 1
        if nz % 2 != 0:
            nz += 1
        
        a, b = x_limits
        c, d = y_limits
        e, f = z_limits
        
        hx = (b - a) / nx
        hy = (d - c) / ny
        hz = (f - e) / nz
        
        x_vals = np.linspace(a, b, nx + 1)
        y_vals = np.linspace(c, d, ny + 1)
        z_vals = np.linspace(e, f, nz + 1)
        
        integral = 0
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    weight = 1
                    
                    for idx, n in [(i, nx), (j, ny), (k, nz)]:
                        if idx == 0 or idx == n:
                            weight *= 1
                        elif idx % 2 == 1:
                            weight *= 4
                        else:
                            weight *= 2
                    
                    integral += weight * func(x_vals[i], y_vals[j], z_vals[k])
        
        return integral * hx * hy * hz / 27
    
    def plot_to_base64(self, fig):
        """Convierte un plot de matplotlib a base64 para mostrar en HTML"""
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_url
    
    # INTEGRALES SIMPLES (1D)
    def area_bajo_curva(self, func_str, a, b, n=1000):
        """Calcula el área bajo una curva"""
        try:
            func_sym = sp.sympify(func_str)
            func = lambdify(self.x, func_sym, 'numpy')
            
            area = self.simpson_1d(func, a, b, n)
            
            # Crear gráfico
            x_plot = np.linspace(a, b, 1000)
            y_plot = func(x_plot)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {func_str}')
            ax.fill_between(x_plot, 0, y_plot, alpha=0.3, color='blue')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=a, color='r', linestyle='--', alpha=0.7, label=f'x = {a}')
            ax.axvline(x=b, color='r', linestyle='--', alpha=0.7, label=f'x = {b}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Área bajo la curva = {area:.6f}')
            ax.legend()
            
            plot_url = self.plot_to_base64(fig)
            
            return {
                'resultado': area,
                'grafico': plot_url,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    def area_entre_curvas(self, func1_str, func2_str, a, b, n=1000):
        """Calcula el área entre dos curvas"""
        try:
            func1_sym = sp.sympify(func1_str)
            func2_sym = sp.sympify(func2_str)
            func1 = lambdify(self.x, func1_sym, 'numpy')
            func2 = lambdify(self.x, func2_sym, 'numpy')
            
            def diff_func(x):
                return abs(func1(x) - func2(x))
            
            area = self.simpson_1d(diff_func, a, b, n)
            
            # Crear gráfico
            x_plot = np.linspace(a, b, 1000)
            y1_plot = func1(x_plot)
            y2_plot = func2(x_plot)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y1_plot, 'b-', linewidth=2, label=f'f₁(x) = {func1_str}')
            ax.plot(x_plot, y2_plot, 'r-', linewidth=2, label=f'f₂(x) = {func2_str}')
            ax.fill_between(x_plot, y1_plot, y2_plot, alpha=0.3, color='green')
            ax.axvline(x=a, color='k', linestyle='--', alpha=0.7)
            ax.axvline(x=b, color='k', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Área entre curvas = {area:.6f}')
            ax.legend()
            
            plot_url = self.plot_to_base64(fig)
            
            return {
                'resultado': area,
                'grafico': plot_url,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    def longitud_arco(self, func_str, a, b, n=1000):
        """Calcula la longitud de arco de una curva"""
        try:
            func_sym = sp.sympify(func_str)
            derivada = diff(func_sym, self.x)
            
            arc_func_sym = sqrt(1 + derivada**2)
            arc_func = lambdify(self.x, arc_func_sym, 'numpy')
            
            longitud = self.simpson_1d(arc_func, a, b, n)
            
            # Crear gráfico
            x_plot = np.linspace(a, b, 1000)
            y_plot = lambdify(self.x, func_sym, 'numpy')(x_plot)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_plot, y_plot, 'b-', linewidth=3, label=f'f(x) = {func_str}')
            ax.axvline(x=a, color='r', linestyle='--', alpha=0.7, label=f'x = {a}')
            ax.axvline(x=b, color='r', linestyle='--', alpha=0.7, label=f'x = {b}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Longitud de arco = {longitud:.6f}')
            ax.legend()
            
            plot_url = self.plot_to_base64(fig)
            
            return {
                'resultado': longitud,
                'grafico': plot_url,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    # INTEGRALES DOBLES (2D)
    def volumen_bajo_superficie(self, func_str, x_limits, y_limits, nx=30, ny=30):
        """Calcula el volumen bajo una superficie z = f(x,y)"""
        try:
            func_sym = sp.sympify(func_str)
            func = lambdify([self.x, self.y], func_sym, 'numpy')
            
            volumen = self.simpson_2d(func, x_limits, y_limits, nx, ny)
            
            # Crear gráfico 3D
            x_plot = np.linspace(x_limits[0], x_limits[1], 40)
            y_plot = np.linspace(y_limits[0], y_limits[1], 40)
            X, Y = np.meshgrid(x_plot, y_plot)
            Z = func(X, Y)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'Volumen bajo la superficie = {volumen:.6f}')
            
            plot_url = self.plot_to_base64(fig)
            
            return {
                'resultado': volumen,
                'grafico': plot_url,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    def area_region_plana(self, x_limits, y_limits, nx=50, ny=50):
        """Calcula el área de una región plana"""
        try:
            def func_const(x, y):
                return 1
            
            area = self.simpson_2d(func_const, x_limits, y_limits, nx, ny)
            
            return {
                'resultado': area,
                'grafico': None,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    def area_superficie(self, func_str, x_limits, y_limits, nx=25, ny=25):
        """Calcula el área de una superficie z = f(x,y)"""
        try:
            func_sym = sp.sympify(func_str)
            
            fx = diff(func_sym, self.x)
            fy = diff(func_sym, self.y)
            
            area_func_sym = sqrt(1 + fx**2 + fy**2)
            area_func = lambdify([self.x, self.y], area_func_sym, 'numpy')
            
            area = self.simpson_2d(area_func, x_limits, y_limits, nx, ny)
            
            return {
                'resultado': area,
                'grafico': None,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    # INTEGRALES TRIPLES (3D)
    def volumen_solido(self, x_limits, y_limits, z_limits, nx=15, ny=15, nz=15):
        """Calcula el volumen de un sólido"""
        try:
            def func_const(x, y, z):
                return 1
            
            volumen = self.simpson_3d(func_const, x_limits, y_limits, z_limits, nx, ny, nz)
            
            return {
                'resultado': volumen,
                'grafico': None,
                'error': None
            }
        except Exception as e:
            return {
                'resultado': None,
                'grafico': None,
                'error': str(e)
            }
    
    def centro_masa(self, densidad_str, x_limits, y_limits, z_limits, nx=12, ny=12, nz=12):
        """Calcula el centro de masa de un sólido con densidad ρ(x,y,z)"""
        try:
            densidad_sym = sp.sympify(densidad_str)
            densidad = lambdify([self.x, self.y, self.z], densidad_sym, 'numpy')
            
            # Masa total
            masa = self.simpson_3d(densidad, x_limits, y_limits, z_limits, nx, ny, nz)
            
            # Momentos
            def momento_x(x, y, z):
                return x * densidad(x, y, z)
            
            def momento_y(x, y, z):
                return y * densidad(x, y, z)
            
            def momento_z(x, y, z):
                return z * densidad(x, y, z)
            
            Mx = self.simpson_3d(momento_x, x_limits, y_limits, z_limits, nx, ny, nz)
            My = self.simpson_3d(momento_y, x_limits, y_limits, z_limits, nx, ny, nz)
            Mz = self.simpson_3d(momento_z, x_limits, y_limits, z_limits, nx, ny, nz)
            
            # Centro de masa
            x_cm = Mx / masa if masa != 0 else 0
            y_cm = My / masa if masa != 0 else 0
            z_cm = Mz / masa if masa != 0 else 0
            
            return {
                'masa': masa,
                'centro_masa': [x_cm, y_cm, z_cm],
                'error': None
            }
        except Exception as e:
            return {
                'masa': None,
                'centro_masa': None,
                'error': str(e)
            }

# Crear instancia del calculador
calc = SimpsonIntegralCalculator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    try:
        data = request.json
        tipo = data['tipo']
        
        if tipo == 'area_bajo_curva':
            resultado = calc.area_bajo_curva(
                data['funcion'],
                float(data['a']),
                float(data['b'])
            )
        
        elif tipo == 'area_entre_curvas':
            resultado = calc.area_entre_curvas(
                data['funcion1'],
                data['funcion2'],
                float(data['a']),
                float(data['b'])
            )
        
        elif tipo == 'longitud_arco':
            resultado = calc.longitud_arco(
                data['funcion'],
                float(data['a']),
                float(data['b'])
            )
        
        elif tipo == 'volumen_bajo_superficie':
            resultado = calc.volumen_bajo_superficie(
                data['funcion'],
                [float(data['xa']), float(data['xb'])],
                [float(data['ya']), float(data['yb'])]
            )
        
        elif tipo == 'area_region_plana':
            resultado = calc.area_region_plana(
                [float(data['xa']), float(data['xb'])],
                [float(data['ya']), float(data['yb'])]
            )
        
        elif tipo == 'area_superficie':
            resultado = calc.area_superficie(
                data['funcion'],
                [float(data['xa']), float(data['xb'])],
                [float(data['ya']), float(data['yb'])]
            )
        
        elif tipo == 'volumen_solido':
            resultado = calc.volumen_solido(
                [float(data['xa']), float(data['xb'])],
                [float(data['ya']), float(data['yb'])],
                [float(data['za']), float(data['zb'])]
            )
        
        elif tipo == 'centro_masa':
            resultado = calc.centro_masa(
                data['densidad'],
                [float(data['xa']), float(data['xb'])],
                [float(data['ya']), float(data['yb'])],
                [float(data['za']), float(data['zb'])]
            )
        
        else:
            resultado = {'error': 'Tipo de cálculo no válido'}
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)