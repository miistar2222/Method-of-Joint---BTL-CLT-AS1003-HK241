import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

'''
first: add data
second: calculating
last: draw truss
'''

###---data processing---###
##---input---##
members = pd.read_excel("inputdata.xlsx", "Member")
joints = pd.read_excel("inputdata.xlsx", "Joint")
forces = pd.read_excel("inputdata.xlsx", "Force")

np.set_printoptions(precision=2, suppress=True) #round to 00 after ','

##-----analysis-----##
#-----joint-----#
joint = {}
for _, row in joints.iterrows():
    joint_coordition = int(row['Joint'])
    x = float(row['X'])
    y = float(row['Y'])
    joint[joint_coordition] = (x, y) #joint coordition 

#-----force-----#
eforce = {} #external force
for _, row in forces.iterrows():
    if row['FX'] != 0 or row['FY'] != 0:
        joint_coordition = int(row['Joint'])
        fx = float(row['FX'])
        fy = float(row['FY'])
        eforce[joint_coordition] = (fx, fy) 

#-----reaction-----#
reaction = {} #reaction
for _, row in forces.iterrows():
   if row['RX'] != 0 or row['RY'] != 0:
        joint_coordition = int(row['Joint'])
        rx = int(row['RX'])
        ry = int(row['RY'])
        reaction[joint_coordition] = (rx, ry)

count_reaction = 0  #count how many reaction in truss
for i in reaction.values():
    count_reaction += i.count(1)
    
#-----members-----#
member = list(zip(members['Start'], members['End'])) #member with the start and end 

count_member = len(member) #counting member

'''
print('input: ')
print('eforce =', eforce)
print('reaction =', reaction)
print('joint =', joint)
print('member =', member)
'''

###-----calculating------###

#-----calculating angle at 2 joint-----#
def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.atan2(dy, dx)
    return angle

'''
AS = B
A is all information about any joint
F is external forces of any joint

--> S = (A^-1) F
using pinv to calculate matrix A inverse
'''

'''
calculating:
step 1: create matrix A and matrix F
step 2: add forces, reaction infor to matrix A
step 3: add external force infor to matrix F
step 4: solve S 
'''

##-----create matrix-----##
m_a = np.zeros((2 * len(joint), count_member + count_reaction), dtype=float)
m_f = np.zeros((2 * len(joint), 1), dtype=float)

##-----add in4 to A-----###
#-----add force-----#
for i, (start, end) in enumerate(member):
    angle = calculate_angle(joint[start], joint[end])
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    #start joint
    m_a[2 * (start - 1), i] += cos_theta
    m_a[2 * (start - 1) + 1, i] += sin_theta

    #end joint
    m_a[2 * (end - 1), i] -= cos_theta
    m_a[2 * (end - 1) + 1, i] -= sin_theta

#-----add reaction-----#
reaction_add = len(member)
for point, (r_x, r_y) in reaction.items():
    if r_x == 1:
        m_a[2 * (point - 1), reaction_add] = 1
        reaction_add += 1
    if r_y == 1:
        m_a[2 * (point - 1) + 1, reaction_add] = 1
        reaction_add += 1

##-----add in4 to F-----##
for point, (f_x, f_y) in eforce.items():
    m_f[2 * (point - 1)][0] += f_x
    m_f[2 * (point - 1) + 1][0] += f_y

##-----solve S-----##
minv_a = np.linalg.pinv(m_a)
m_s = np.dot(minv_a, -m_f)


##-----answers-----##

print('Forces:')
for i, (start, end) in enumerate(member):
    print(f'Bar {start}-{end}: {m_s[i][0]:.2f} ({"Tension" if m_s[i][0] > 0 else "Compression" if m_s[i][0] < 0 else "None"})')
    
print('\nExternal Forces:')
for point, force in eforce.items():
    print(f'Point {point}: FX = {force[0]:.2f}, FY = {force[1]:.2f}')

print('\nReactions:')
reaction_values = m_s[count_member:]
index = 0

for point, (rx, ry) in reaction.items():
    if rx == 1:
        print(f'Point {point}: RX = {reaction_values[index][0]:.2f}')
        index += 1
    if ry == 1:
        print(f'Point {point}: RY = {reaction_values[index][0]:.2f}')
        index += 1

###-----draw-----###
fig, ax = plt.subplots()  

#-----draw button-----#
for j, (x, y) in joint.items():
    ax.plot(x, y, 'o', color='black')
    ax.text(x, y, f'{j}', color='black', fontsize=16, ha='right')

#-----color-----#
def get_color(force):
    if force > 0:
        return '#fe6d7a'   #Tension
    elif force < 0:
        return '#2caddb'   #Compression
    else:
        return '#6b6b6b'   #None

#-----draw trusses-----#
for i, (start, end) in enumerate(member):

    x_coor = [joint[start][0], joint[end][0]]
    y_coor = [joint[start][1], joint[end][1]]

    ax.plot(x_coor, y_coor, color=get_color(m_s[i][0]), linewidth=2)

    #write answers
    mid_x = (joint[start][0] + joint[end][0]) / 2
    mid_y = (joint[start][1] + joint[end][1]) / 2
    ax.text(mid_x, mid_y, f'{m_s[i][0]:.2f}', fontsize=10)
 
#-----draw external force vectors-----#
for point, (fx, fy) in eforce.items():
    x, y = joint[point]
    # Draw FX
    if fx != 0:
        fxa = -1 if fx < 0 else 1
        ax.quiver(x, y, fxa, 0, angles='xy', scale_units='xy', scale=1, color='#4A26AB', width=0.005)
        ax.text(x + 0.1 * fxa, y, f'FX = {fx:.2f}', fontsize=10, ha='left')

    # Draw FY
    if fy != 0:
        fya = -1 if fy < 0 else 1
        ax.quiver(x, y, 0, fya, angles='xy', scale_units='xy', scale=1, color='#4A26AB', width=0.005)
        ax.text(x, y + 0.1 * fya, f'FY = {fy:.2f}', fontsize=10, ha='left')
        
#-----draw reaction force vectors-----#
index = count_member

for point, (rx, ry) in reaction.items():
    x, y = joint[point]

    #check for direction of RX
    if rx == 1:

        rxa = -1 if m_s[index][0] < 0 else 1
        ax.quiver(x, y, rxa, 0, angles='xy', scale_units='xy', scale=1, color='#399407', linewidth=1.5)
        ax.text(x + 0.1 * rxa, y, f'{m_s[index][0]:.2f}', fontsize=10, ha='center', va='center')
        index += 1

    #check for direction of Ry
    if ry == 1:

        rya = -1 if m_s[index][0] < 0 else 1
        ax.quiver(x, y, 0, rya, angles='xy', scale_units='xy', scale=1, color='#399407', linewidth=1.5)
        ax.text(x, y + 0.1 * rya, f'{m_s[index][0]:.2f}', fontsize=10, ha='center', va='center')
        index += 1



#-----legend description-----#
ax.scatter([], [], color='#fe6d7a', label='Tension')
ax.scatter([], [], color='#2caddb', label='Compression')
ax.scatter([], [], color='#6b6b6b', label='None')
ax.scatter([], [], color='#4A26AB', label='External force')
ax.scatter([], [], color='#399407', label='Reaction')

ax.legend()

'''
hex corlor(description nearly):
#fe6d7a red
#2caddb blue
#6b6b6b gray
#4A26AB purple
#399407 green
'''

#-----setting-----#
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Trusses using Method of joint")
ax.grid(True)
ax.axis("equal")

plt.show()
