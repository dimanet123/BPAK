import math
import openpyxl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pnd
import numpy as np

mu1 = 398600.79
mu2 = 4920.8
eps = 26327850000.0
r = 6878.137
pi = 6.283185307/2



final = openpyxl.Workbook()  # Создание нового расписания с именем final
sheet = final.create_sheet("ControlPoints")

h = 10  # Шаг
x0 = 0  # Начальное значение x
y0 = 0  # Начальное значение y
z0 = 7.35336563078044  # Начальное значение y'

x1 = 0  # Начальное значение x
y1 = r  # Начальное значение y
z1 = 0  # Начальное значение y'

x2 = 0  # Начальное значение x
y2 = 0  # Начальное значение y
z2 = 1.97032838241839  # Начальное значение y'
n = 6000  # Количество итераций
dopy = 0
y=0
def update(Planet,s):
    ν = Planet.V/Planet.a*s
    a = Planet.a
    e = Planet.e
    i = Planet.i  # Наклонение
    Ω = Planet.Ω  # Долгота восходящего узла
    ω = Planet.ω
    r = a * (1 - e**2) / (1 + e * np.cos(ν))
    X = r * (np.cos(Ω) * np.cos(ω + ν) - np.sin(Ω) * np.sin(ω + ν) * np.cos(i))
    Y = r * (np.sin(Ω) * np.cos(ω + ν) + np.cos(Ω) * np.sin(ω + ν) * np.cos(i))
    Z = r * np.sin(ω + ν) * np.sin(i)
    Planet.update_position(X, Y, Z)
    Planet.ω = ω + ν

class Point:
    def __init__(self, t, x, y, z, vx, vy, vz, mu,radius,u):
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mu = mu
        self.radius = radius
        self.u = u
        
class Planet:
    def __init__(self, x, y, z, mu, radius, a, e, i, Ω, ω, V):
        self.x = x
        self.y = y
        self.z = z
        self.mu = mu
        self.radius = radius
        self.a = a
        self.e = e
        self.i = i
        self.Ω = Ω
        self.ω = ω
        self.history = []
        self.save_state()
        self.V = V
        
    def save_state(self):
        state = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'mu': self.mu,
            'radius': self.radius,
            'a': self.a,
            'e': self.e,
            'i': self.i,
            'Ω': self.Ω,
            'ω': self.ω,
            # 'V': self.V
        }
        self.history.append(state)

    def update_position(self, new_x, new_y, new_z):
        self.x = new_x
        self.y = new_y
        self.z = new_z
        self.save_state()

    def get_history_as_dataframe(self):
        return pnd.DataFrame(self.history)
        
def distance(point1, point2):
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    dz = point1.z - point2.z
    return math.sqrt(dx**2 + dy**2 + dz**2)

Sun = Planet(0, 0, 0, 132712440018, 696340, 0.000000000001,0,0,0,9,0)
Earth = Planet(0, 0, 0, 398600.79, 6878.137, 150000000,0.0167,np.radians(30),-np.radians(90),0,27.79)
Mercury = Planet(0, 0, 0, 398600.79, 6878.137, 150000000,0.0167,np.radians(45),-np.radians(90),0,27.79)
Earth3 = Planet(0, 0, 0, 398600.79, 6878.137, 150000000,0.0167,np.radians(60),-np.radians(90),0,27.79)
Planets = [Sun,Earth,Earth2,Earth3]
def rh(x):
    result = Point(1, x.vx, x.vy, x.vz, 0, 0, 0, 0,0,0)
    for planet in Planets:
        r = distance(x, planet)
        result.vx += -planet.mu * (x.x - planet.x) / r ** 3
        result.vy += -planet.mu * (x.y - planet.y) / r ** 3
        result.vz += -planet.mu * (x.z - planet.z) / r ** 3
    v = math.sqrt(x.vx ** 2 + x.vy ** 2 + x.vz ** 2)
    result.u = math.sqrt((x.y * x.vz - x.z * x.vy) ** 2 + (x.z * x.vx - x.x * x.vz) ** 2 + (x.x * x.vy - x.y * x.vx) ** 2) / r ** 2

    return result

def rkstep(x, s, deltaV):
    x.vx += deltaV[0]
    x.vy += deltaV[1]
    x.vz += deltaV[2]
    h1 = rh(x)
    y = Point(x.t + 0.5 * s * h1.t, x.x + 0.5 * s * h1.x, x.y + 0.5 * s * h1.y, x.z + 0.5 * s * h1.z,
              x.vx + 0.5 * s * h1.vx, x.vy + 0.5 * s * h1.vy, x.vz + 0.5 * s * h1.vz,0,0, x.u + 0.5 * s * h1.u)

    h2 = rh(y)
    y.t = x.t + 0.5 * s * h2.t
    y.x = x.x + 0.5 * s * h2.x
    y.y = x.y + 0.5 * s * h2.y
    y.z = x.z + 0.5 * s * h2.z
    y.vx = x.vx + 0.5 * s * h2.vx
    y.vy = x.vy + 0.5 * s * h2.vy
    y.vz = x.vz + 0.5 * s * h2.vz
    y.u = x.u + 0.5 * s * h2.u

    h3 = rh(y)
    y.t = x.t + s * h3.t
    y.x = x.x + s * h3.x
    y.y = x.y + s * h3.y
    y.z = x.z + s * h3.z
    y.vx = x.vx + s * h3.vx
    y.vy = x.vy + s * h3.vy
    y.vz = x.vz + s * h3.vz
    y.u = x.u + s * h3.u

    h4 = rh(y)

    y.t = x.t + s * (h1.t + 2 * h2.t + 2 * h3.t + h4.t) / 6
    y.x = x.x + s * (h1.x + 2 * h2.x + 2 * h3.x + h4.x) / 6
    y.y = x.y + s * (h1.y + 2 * h2.y + 2 * h3.y + h4.y) / 6
    y.z = x.z + s * (h1.z + 2 * h2.z + 2 * h3.z + h4.z) / 6
    y.vx = x.vx + s * (h1.vx + 2 * h2.vx + 2 * h3.vx + h4.vx) / 6
    y.vy = x.vy + s * (h1.vy + 2 * h2.vy + 2 * h3.vy + h4.vy) / 6
    y.vz = x.vz + s * (h1.vz + 2 * h2.vz + 2 * h3.vz + h4.vz) / 6
    y.u = x.u + s * (h1.u + 2 * h2.u + 2 * h3.u + h4.u) / 6

    return y


def orbit(h,i):
    Re = Sun.radius
    v = math.sqrt(Sun.mu/(Re+h))
    r=Re + h
    vz= v * math.sin(i * pi /180)
    vy = v  * math.cos(i * pi /180)
    result = pnd.DataFrame(columns=['time','x','y','z','Vx','Vy','Vz'])
    s=10000
    data = ["t","u","vx","vy","vz","x","y","z"]
    sheet.append(data)
    x = Point(0,0,-r,0,vy,0,0,0,0,0)
    data = [x.t,x.u,x.vx,x.vy,x.vz,x.x,x.y,x.z]
    sheet.append(data)
    for _ in range(6000):
        for planet in Planets:
            update(planet, s)
        deltaV = [0,0,0]
        # if _ == 32:
        #     deltaV = 2.9
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # elif _ == 960:
        #     deltaV = -3.33
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # elif _ == 1200:
        #     deltaV = 0.2
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # elif _ == 1500:
        #     deltaV = -0.3
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # elif _ == 4000:
        #     deltaV =-0.4
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # elif _ == 550:
        #     deltaV =-0.2
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # elif _ == 650:
        #     deltaV =-0.2
        #     vector = [result.loc[len(result)-1,'x'] - result.loc[len(result)-2,'x'], result.loc[len(result)-1,'y'] - result.loc[len(result)-2,'y'],result.loc[len(result)-1,'z'] - result.loc[len(result)-2,'z']]
        #     top = math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        #     vector[0] = vector[0]*deltaV/top
        #     vector[1] = vector[1]*deltaV/top
        #     vector[2] = vector[2]*deltaV/top
        #     x = rkstep(x, s,vector)
        # else: 
        x = rkstep(x, s,deltaV)
        data = [x.t,x.x,x.y,x.z,x.vx,x.vy,x.vz]
        result.loc[len(result)] = data
        sheet.append(data)
    data = [x.t,x.u,x.vx,x.vy,x.vz,x.x,x.y,x.z]
    sheet.append(data)
    return result
data = orbit(149500000, 0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Sun.x, Sun.y , Sun.z, s=300, c = 'yellow', label=f"Солнце")
# ax.scatter(Earth.x, Earth.y , Earth.z, s=100, label=f"земля position")
# df = Earth.get_history_as_dataframe()
for planet in Planets:
    ax.scatter(planet.x, planet.y , planet.z, s=100, label=f"земля position{planet.i}")
    df = planet.get_history_as_dataframe()
    ax.scatter(df.x, df.y, df.z, label='Группа 1', c='b', marker='.')
    
x1 = data['x']
y1 = data['y']
z1 = data['z']
max_range = np.array([x1, y1, z1]).ptp(axis=1).max() / 2.0

mid_x = (np.array(x1).max() + np.array(x1).min()) * 0.5
mid_y = (np.array(y1).max() + np.array(y1).min()) * 0.5
mid_z = (np.array(z1).max() + np.array(z1).min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.scatter(x1, y1, z1, label='Группа 1', c='r', marker='.')
# ax.scatter(df.x, df.y, df.z, label='Группа 1', c='b', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.legend()
final.save('CalendarPlan_2023_2024.xlsx')
plt.show()

