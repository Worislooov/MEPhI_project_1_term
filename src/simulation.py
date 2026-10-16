"""
Модуль моделирования траектории у чёрной дыры.
"""
from scipy.integrate import solve_ivp
import numpy as np



class PhotonSimulation:


    def __init__(self, r_s: float = 1.0, b: float = 5.0, phi_0: float = 1e-3, phi_max: float = 8 * np.pi):
        """
        Начальные параметры.
        """
        assert r_s > 0
        assert b > 0
        assert 0 < phi_0 < np.pi

        self.r_s = float(r_s)
        self.b = float(b)
        self.phi_0 = float(phi_0)
        self.phi_max = float(phi_max)
        self.u = None
        self.phi = None
        self.r = None
        self.x = None
        self.y = None

        self.status = "Старта еще не было"
        self.r_min = None

    def _ode_systemus(self, phi: float, state: list):
        """
        Система ДУ.
        """
        u, du_dphi = state

        d2u_dphi2 = 1.5 * self.r_s * (u ** 2) - u

        return [du_dphi, d2u_dphi2]

    def _event_horizont(self, phi: float, state: list):
        u = state[0]
        return (1.0 / self.r_s) - u

    # если фотон пересек горизонт
    _event_horizont.terminal = True
    _event_horizont.direction = -1

    
    def _event_esc(self, phi: float, state: list):
    #Если фотон на безопасной дистанции
        u = state[0]
        initial_u = np.sin(self.phi_0) / self.b
        return u - initial_u * 0.5

    _event_esc.terminal = True
    _event_esc.direction = -1

    def run(self, rtol: float = 1e-8, atol: float = 1e-10, max_step: float = 0.05):

        # старт луча (точка откуда стартуем)
        u_0 = np.sin(self.phi_0) / self.b
        du_dphi_0 = np.cos(self.phi_0) / self.b
        initial_state = [u_0, du_dphi_0]

        # здесь метод рк45
        sol = solve_ivp(
            fun=self._ode_systemus,
            t_span=(self.phi_0, self.phi_max),
            y0=initial_state,
            method='RK45',
            events=[self._event_horizont, self._event_esc],
            rtol=rtol,
            atol=atol,
            max_step=max_step
        )

        self.phi = sol.t
        self.u = sol.y[0]

        assert np.all(self.u > 0), "U стала отрицательной!"

        # перевод чисел в декартовую
        self.r = 1.0 / self.u
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)
        self.r_min = float(np.min(self.r))

        if len(sol.t_events[0]) > 0:
            self.status = "В черной дыре"
        elif len(sol.t_events[1]) > 0 or self.phi[-1] >= self.phi_max:
            self.status = "Далеко от черной дыры"
        else:
            self.status = "Обогнул и вылетел"

        return self.phi, self.u, self.r, self.x, self.y
