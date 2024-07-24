import numpy as np


def back_diff(x: np.ndarray, t: np.array, order: int = 4, var: bool = False):
    dx = np.zeros((len(x), 3), dtype=float)
    dt = np.insert(t[1:] - t[:-1], 0, 0)

    if order == 4:
        # 4th Order Backward Finite Difference
        if var:
            # Differentiation w/ varying dt
            for i in range(4, len(t)):
                h1 = t[i] - t[i - 1]
                h2 = t[i] - t[i - 2]
                h3 = t[i] - t[i - 3]
                h4 = t[i] - t[i - 4]

                a = (h1 * (h2 * (h3 + h4) + h3 * h4) + h2 * h3 * h4) / (h1 * h2 * h3 * h4)
                b = h2 * h3 * h4 / (h1 * (h1 - h4) * (h1 - h3) * (h1 - h2))
                c = -h1 * h3 * h4 / (h2 * (h1 - h2) * (h2 - h4) * (h2 - h3))
                d = h1 * h2 * h4 / (h3 * (h1 - h3) * (h2 - h3) * (h3 - h4))
                e = -h1 * h2 * h3 / (h4 * (h1 - h4) * (h2 - h4) * (h3 - h4))

                dx[i, :] = a * x[i, :] + b * x[i - 1, :] + c * x[i - 2, :] + d * x[i - 3, :] + e * x[i - 4, :]
            return dx

        if not var:
            # 4th Order Backward Finite Difference Coefficients
            den = 12
            num = [3, -16, 36, -48, 25]
            for i in range(3, len(x)):
                dx[i, :] = (num[0] * x[i - 4, :] + num[1] * x[i - 3, :] + num[2] * x[i - 2, :] + num[3] * x[i - 1, :] +
                            num[4] * x[i, :]) / (den * np.average(dt[i-4:i+1]))

    if order == 6:
        # 6th Order Central Finite Difference
        den = 60
        num = [10, -72, 225, -400, 450, -360, 147]
        for i in range(5, len(x)):
            dx[i, :] = (num[0] * x[i - 6, :] + num[1] * x[i - 5, :] + num[2] * x[i - 4, :] + num[3] * x[i - 3, :] +
                        num[4] * x[i - 2, :] + num[5] * x[i - 1, :] + num[6] * x[i, :]) / (den * np.average(dt[i-6:i+1]))

    if order == 8:
        # 8th Order Backward Finite Difference
        den = 3.748374423901926e+82
        num = [1.0187546202093276e+83, -2.9986995390864937e+83, 5.247724193324426e+83, -6.996965591015617e+83,
               6.559655241526099e+83, -4.1981793545536496e+83, 1.7492413977236302e+83, -4.2838564842079276e+82,
               4.685468029591042e+81]
        for i in range(7, len(x)):
            dx[i, :] = (num[0] * x[i, :] + num[1] * x[i - 1, :] + num[2] * x[i - 2, :] + num[3] * x[i - 3, :] +
                        num[4] * x[i - 4, :] + num[5] * x[i - 5, :] + num[6] * x[i - 6, :] + num[7] * x[i - 7, :] +
                        num[8] * x[i - 8, :]) / (den * np.average(dt[i-8:i+1]))
    return dx


def central_diff(x: np.ndarray, t: np.array, order: int = 4):
    dx = np.zeros((len(x), 3), dtype=float)
    dt = np.insert(t[1:] - t[:-1], 0, 0)

    if order == 4:
        # 4th Order Central Finite Difference
        den = 12
        num = [1, -8, 0, 8, -1]
        for i in range(2, len(x)-2):
            dx[i, :] = (num[0] * x[i - 2, :] + num[1] * x[i - 1, :] + num[2] * x[i, :] + num[3] * x[i + 1, :] +
                        num[4] * x[i + 2, :]) / (den * np.average(dt[i-2:i+3]))

    if order == 6:
        # 6th Order Central Finite Difference
        den = 60
        num = [-1, 9, -45, 0, 45, -9, 1]
        for i in range(3, len(x)-3):
            dx[i, :] = (num[0] * x[i - 3, :] + num[1] * x[i - 2, :] + num[2] * x[i - 1, :] + num[3] * x[i, :] +
                        num[4] * x[i + 1, :] + num[5] * x[i + 2, :] + num[6] * x[i + 3, :]) / (den * np.average(dt[i-3:i+4]))

    if order == 8:
        # 8th Order Central Finite Difference
        den = 840
        num = [3, -32, 168, -672, 0, 672, -168, 32, -3]
        for i in range(4, len(x)-4):
            dx[i, :] = (num[0] * x[i - 4, :] + num[1] * x[i - 3, :] + num[2] * x[i - 2, :] + num[3] * x[i - 1, :] +
                        num[4] * x[i, :] + num[5] * x[i + 1, :] + num[6] * x[i + 2, :] + num[7] * x[i + 3, :] +
                        num[8] * x[i + 4, :]) / (den * np.average(dt[i-4:i+5]))
    return dx
