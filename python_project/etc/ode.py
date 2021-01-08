import numpy as np

g = 9.8
l = 2
mu = 0.1

Theta_0 = np.pi / 3
Theta_dot = 0


def get_theta_doube_dot(theta, theta_dot):
    return -mu * theta_dot - (g / l) * np.sin(theta)


def theta(t):
    theta = Theta_0
    theta_dot = Theta_dot
    delta_t = 0.01
    for time in np.array(0, t, delta_t):
        theta_double_dot = get_theta_doube_dot(theta, theta_dot)
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
    return theta
