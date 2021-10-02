p = [14708874.78, 16480015.63, 18889106.77]
s2024 = [0.2197, 0.1604]
w2024 = [0.0457, 0.0392]
u2024 = [0.0746, 0.0248]
s2030 = [0.1099, 0.0098, 0]
w2030 = [0.0317, 0.0159, 0]
u2030 = [0, 0, 0]
D2019 = [6296.568, 30620.68]

s_aversion_2024_s1 = (p[1]/p[0])*((0.3132-s2024[0])/0.37)*D2019[0]
s_aversion_2024_s2 = (p[1]/p[0])*((0.3132-s2024[1])/0.37)*D2019[0]
w_aversion_2024_s1 = (p[1]/p[0])*((0.05-w2024[0])/0.07)*D2019[1]
w_aversion_2024_s2 = (p[1]/p[0])*((0.05-w2024[1])/0.07)*D2019[1]

s_aversion_2030_s1 = (p[2]/p[0])*((0.2479-s2030[0])/0.37)*D2019[0]
s_aversion_2030_s2 = (p[2]/p[0])*((0.2479-s2030[1])/0.37)*D2019[0]
# w_aversion_2030_s1 = (p[2]/p[0])*((0.05-w2030[0])/0.07)*D2019[1]
# w_aversion_2030_s2 = (p[2]/p[0])*((0.05-w2030[1])/0.07)*D2019[1]

print("Stunting aversion-2024")
print(s_aversion_2024_s1)
print(s_aversion_2024_s2)
print("Wasting aversion-2024")
print(w_aversion_2024_s1)
print(w_aversion_2024_s2)

print("Stunting aversion-2030")
print(s_aversion_2030_s1)
print(s_aversion_2030_s2)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Stunted_2024_P1")
print(0.2197 * p[1])
print("Wasted_2024_P1")
print(0.0457 * p[1])
print("Under_weight_2024_P1")
print(0.0746 * p[1])
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Stunted_2024_P2")
print(0.1604 * p[1])
print("Wasted_2024_P2")
print(0.0392 * p[1])
print("Under_weight_2024_P2")
print(0.0248 * p[1])
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Stunted_2030_P1")
print(0.1099 * p[2])
print("Wasted_2030_P1")
print(0.0317 * p[2])
print("Under_weight_2030_P1")
print(0.0 * p[2])
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Stunted_2030_P2")
print(0.0098 * p[2])
print("Wasted_2030_P2")
print(0.0159 * p[2])
print("Under_weight_2030_P2")
print(0.0 * p[2])
# print("Wasting aversion-2030")
# print(w_aversion_2030_s1)
# print(w_aversion_2030_s2)