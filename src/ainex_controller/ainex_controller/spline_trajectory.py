import numpy as np

def polynomial_3rd_order(xi: float, xf: float, vi: float, vf: float, 
						 ti: float, tf: float) -> np.ndarray:
	x = np.array([xi, xf, vi, vf])
	T = np.array([[1, ti, ti**2, ti**3],
				  [1, tf, tf**2, tf**3],
				  [0, 1, 2*ti, 3*ti**2],
				  [0, 1, 2*tf, 3*tf**2]])
	a = np.linalg.solve(T, x)
	return a

class LinearSplineTrajectory:
    def __init__(self, x_init: np.ndarray, x_final: np.ndarray, duration: float,
                 v_init: np.ndarray = np.zeros(3), v_final: np.ndarray = np.zeros(3)
                 ) -> None:
        self.x_init = x_init
        self.x_final = x_final
        self.x_coeffs = [np.zeros(4) for _ in range(3)]

        for i in range(3):
            self.x_coeffs[i] = polynomial_3rd_order(x_init[i], x_final[i], v_init[i], v_final[i], 0.0, duration)

    def update(self, t: float) -> np.ndarray:
        x_des = np.zeros(3)
        v_des = np.zeros(3)

        for i in range(3):
            x_des[i] = np.dot(self.x_coeffs[i], np.array([1, t, t**2, t**3]))
            v_des[i] = np.dot(self.x_coeffs[i], np.array([0, 1, 2*t, 3*t**2]))

        return x_des, v_des
