from sympy import *
t1,t2,t3,t4 = symbols("t1,t2,t3,t4", real=True)
L1, L2 = symbols("L1, L2", real=True)
x_elbow,y_elbow,z_elbow = symbols("x_elbow, y_elbow, z_elbow", real=True)
x_elbow,y_elbow,z_elbow = symbols("x_wrist, y_wrist, z_wrist", real=True)


T0_1 = Matrix([
    [cos(t1), 0 , -sin(t1), 0],
    [0, 1, 0, 0],
    [sin(t1), 0, cos(t1), 0],
    [0, 0, 0, 1]
])

T1_2 = Matrix([
    [1, 0, 0 ,0],
    [0, cos(t2), sin(t2), 0],
    [0, -sin(t2), cos(t2), 0],
    [0, 0, 0, 1]
])

T_elbow = Matrix([
    [0 ],
    [L1],
    [0],
    [1]
])

T2_3 = Matrix([
    [cos(t3), 0 , -sin(t3), 0],
    [0, 1, 0, L1],
    [sin(t3), 0, cos(t3), 0],
    [0, 0, 0, 1]
])
T3_4 = Matrix([
    [cos(t4), sin(t4), 0, 0],
    [-sin(t4), cos(t4), 0, 0],
    [0, 0, 1 ,0 ],
    [0, 0, 0, 1]
])

T_wrist = Matrix([
    [0],
    [L2],
    [0],
    [1]
])

pprint("left_arm")

T0_2 = T0_1*T1_2*T_elbow
pprint(T0_2)

T0_wrist = T0_1*T1_2*T2_3*T3_4*T_wrist 
pprint(simplify(T0_wrist))
exit()
t1_sol = atan2(x_elbow, -z_elbow)
t2_sol = atan2(sqrt(x_elbow**2 + z_elbow**2), y_elbow)

p0_wrist_sub = T0_wrist.subs({
    t1: t1_sol,
    t2: t2_sol
})

#pprint(simplify(p0_wrist_sub))



eqs = [
    T0_wrist[0],
    T0_wrist[1],
    T0_wrist[2]
]
#pprint(eqs)
eqs_v2 = [eqs[0], eqs[1], eqs[2]]
sol_wrist = solve(eqs_v2, (t3, t4), dict=True)
pprint(sol_wrist)
#pprint("simplified sol_wrist", simplify(sol_wrist))
# # Elbow
# t1 = atan2(x, z)
# t2 = acos(y/L1)

# # Wrist
# t4 = acos((y - L1*cos(t2)) / L2)
# t3 = atan2(
#     x*cos(t1) - z*sin(t1),
#     x*sin(t1) + z*cos(t1)
# )