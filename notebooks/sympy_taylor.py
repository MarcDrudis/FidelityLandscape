from sympy import Function, O, diff, pprint, series, symbols

t = symbols("t")
theta = symbols("theta")

f = Function("f")(theta, t)

f_taylor = f.series(theta, 0, 4)

f_taylor = f_taylor.subs(O(theta**4), diff(f, theta, 4) / 24)

f_taylor = f_taylor.series(t, 0, 2)

f_taylor = f_taylor.subs(O(t**2), diff(f, t, 2) / 2)

f_taylor = f_taylor.expand()

f_taylor = f_taylor.subs(f.subs([(theta, 0), (t, 0)]), 0)


pprint(f_taylor)
