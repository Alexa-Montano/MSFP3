"""
Práctica 3: Sistema Musculoesquelético

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Alexa Montano Gomez 
Número de control: 23212215
Correo institucional: L23212215@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Librerias
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import pandas as pd
from scipy import signal

# Datos de simulacion
x0, t0, tend, dt, w, h = 0, 0, 15, 1E-3, 7, 3.5

n = round((tend - t0) / dt) + 1
t = np.linspace(t0, tend, n)

# SeÑal de entrada F(t): pulso entre 1 y 2 segundos
F = np.where((t >= 1) & (t < 2), 0.75, 0.0)

# Funcion de transferencia
def musculo(alpha, Cs, Cp, R):
    num = [(Cs * R) + (1 - alpha)]
    den = [R * (Cp + Cs), 1]
    sys = ctrl.tf(num, den)
    return sys

alpha, Cs, Cp, R = 0.25, 10e-6, 100e-6, 10e3
sysCaso = musculo(alpha, Cs, Cp, R)

print('Caso:', sysCaso)

# Respuesta F(t)
respuesta = ctrl.forced_response(
    sysCaso,
    T=t,
    U=F,
    X0=x0
)
Fsy = np.asarray(respuesta.outputs).squeeze()
Fsx = F.copy()             # Control CE
PID = np.where((t >= 1) & (t < 2), 1.0, 0.0)

# Colores
clr1 = np.array([255, 60, 110]) / 255
clr2 = np.array([255, 140, 0]) / 255
clr3 = np.array([0, 120, 255]) / 255
clr4 = np.array([0, 205, 100]) / 255

# Grafica Caso y control
fg1 = plt.figure(figsize=(w, h))
plt.plot(t, Fsx, '-', color=clr1, linewidth=0.75,label='Control')
plt.plot(t, Fsy, '-', color=clr2, linewidth=0.75,label=r'$F(t)$: Caso FT')
plt.plot(t, F, '--', color=clr3, linewidth=0.75,label=r'$F(t)$: Caso CE')
plt.plot(t, PID, ':', color=clr4, linewidth=0.75,label=r'$PID(t)$: Caso')
plt.xlim(0, 10)
plt.xticks(np.arange(0, 11, 1))
plt.ylim(-0.1, 1.1)
plt.yticks(np.arange(-0.1, 1.11, 0.1))
plt.xlabel(r'$t$ [s]', fontsize=11)
plt.ylabel(r'$F(t)$ [V]', fontsize=11)
plt.grid(False)
plt.tick_params(direction='in', top=True, right=True)
plt.legend(loc='lower center',bbox_to_anchor=(0.5, 1.06),ncol=4,frameon=False)
plt.tight_layout()
fg1.savefig('Caso_control_py.pdf', dpi=300, bbox_inches='tight')
plt.show()