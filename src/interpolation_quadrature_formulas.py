# pip install numpy scipy

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import numpy as np
from scipy import special

# Параметры задачи
A_X = 0.0
B_X = 3.0
H_X = 0.3

A_T = 0.0
B_T = math.pi

EPSILON = 1e-6

x_points = np.round(np.arange(A_X, B_X + H_X / 2, H_X), 10)


def g(t, x):
    return np.cos(x * np.cos(t))


def j0_exact(x):
    return special.j0(x)


def right_rect(x_val, N):
    h = (B_T - A_T) / N
    t = np.linspace(A_T + h, B_T, N)
    return (1 / math.pi) * np.sum(g(t, x_val)) * h


def mid_rect(x_val, N):
    h = (B_T - A_T) / N
    t = np.linspace(A_T + h / 2, B_T - h / 2, N)
    return (1 / math.pi) * np.sum(g(t, x_val)) * h


def simpson(x_val, N):
    if N % 2 != 0:
        N += 1
    h = (B_T - A_T) / N
    t_left = np.linspace(A_T, B_T - h, N)
    t_mid = t_left + h / 2
    t_right = t_left + h
    integral = np.sum(g(t_left, x_val) + 4 * g(t_mid, x_val) + g(t_right, x_val)) * h / 6
    return (1 / math.pi) * integral


def gauss2(x_val, N):
    h = (B_T - A_T) / N
    t_left = np.linspace(A_T, B_T - h, N)
    gp1 = t_left + (h / 2) * (1 - 1 / math.sqrt(3))
    gp2 = t_left + (h / 2) * (1 + 1 / math.sqrt(3))
    integral = np.sum(g(gp1, x_val) + g(gp2, x_val)) * h / 2
    return (1 / math.pi) * integral


def compute_eps(method, x_val, eps=EPSILON, N_start=2, max_iter=30):
    N = N_start
    prev = method(x_val, N)
    for _ in range(max_iter):
        N *= 2
        curr = method(x_val, N)
        if abs(curr - prev) < eps:
            return curr, N
        prev = curr
    return curr, N


def print_method(name, method, fixed_n=None):
    print(f"\n{'=' * 75}")
    print(f"  {name}")
    print(f"{'=' * 75}")
    print(f"{'x':>6} | {'J0(x) эталон':>14} | {'J0(x) метод':>14} | {'Погрешность':>12} | {'N':>6}")
    print(f"{'-' * 75}")

    errors = []
    ns = []

    for x_val in x_points:
        exact = j0_exact(x_val)
        if fixed_n is not None:
            val = method(x_val, fixed_n)
            N_used = fixed_n
        else:
            val, N_used = compute_eps(method, x_val)

        err = abs(val - exact)
        errors.append(err)
        ns.append(N_used)

        print(f"{x_val:>6.1f} | {exact:>14.9f} | {val:>14.9f} | {err:>12.2e} | {N_used:>6}")

    print(f"\n  Макс. погрешность: {max(errors):.2e}")
    print(f"  Среднее N:         {np.mean(ns):.0f}")
    return errors, ns


print("ВЫЧИСЛЕНИЕ ФУНКЦИИ БЕССЕЛЯ J0(x)")
print("J0(x) = (1/pi) int_0^pi cos(x*cos(t)) dt")
print(f"Параметры: a=0, b=3, h=0.3, eps=1e-6")

e1, n1 = print_method("1. Правые прямоугольники (N=1024)", right_rect, fixed_n=1024)
e2, n2 = print_method("2. Центральные прямоугольники", mid_rect)
e3, n3 = print_method("3. Формула Симпсона", simpson)
e4, n4 = print_method("4. Формула Гаусса с 2 узлами", gauss2)

print(f"\n{'=' * 55}")
print("  СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
print(f"{'=' * 55}")
print(f"{'Метод':<30} {'Среднее N':>10} {'Макс. погрешность':>17}")
print(f"{'-' * 55}")

summary = [
    ("Правые прямоугольники", e1, n1),
    ("Центральные прямоугольники", e2, n2),
    ("Симпсон", e3, n3),
    ("Гаусс с 2 узлами", e4, n4),
]

for name, errors, ns in summary:
    print(f"{name:<30} {np.mean(ns):>10.0f} {max(errors):>17.2e}")
