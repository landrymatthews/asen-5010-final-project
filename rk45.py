class RK45Integrator:
    def __init__(self, fun, t_span, y0, max_step=np.inf, rtol=1e-3, atol=1e-6):
        self.fun = fun
        self.t_start, self.t_end = t_span
        self.y = np.array(y0, dtype=np.float64)
        self.t = self.t_start
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.h = min(0.1*(self.t_end - self.t_start), self.max_step) # Initial step size

        self.A = np.array([
            [0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        ], dtype=np.float64)

        self.b4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 0], dtype=np.float64)
        self.b5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 1/200], dtype=np.float64)
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1], dtype=np.float64)

    def step(self):
      
        k = np.empty((6, len(self.y)), dtype=np.float64)
        for i in range(6):
            k[i] = self.fun(self.t + self.c[i] * self.h, self.y + self.h * np.dot(self.A[i, :i], k[:i]))

        y_new_4 = self.y + self.h * np.dot(self.b4, k)
        y_new_5 = self.y + self.h * np.dot(self.b5, k)
        
        local_error = np.linalg.norm(y_new_5 - y_new_4)
        
        tolerance = self.atol + self.rtol * np.maximum(np.abs(self.y), np.abs(y_new_5))
        
        error_ratio = local_error / np.linalg.norm(tolerance)
        
        if error_ratio > 1:
            self.h *= 0.9 * error_ratio**(-0.25)
            return False # Reject step

        self.y = y_new_5
        self.t += self.h
        self.h *= 0.9 * error_ratio**(-0.2)
        self.h = min(self.h, self.max_step, self.t_end - self.t)
        return True # Accept step

    def solve(self):
        t_values = [self.t_start]
        y_values = [self.y.copy()]

        while self.t < self.t_end:
            success = self.step()
            if success:
                t_values.append(self.t)
                y_values.append(self.y.copy())
            if self.h < 1e-10:
                print("Warning: step size is very small, integration may not converge.")
                break

        return np.array(t_values), np.array(y_values)