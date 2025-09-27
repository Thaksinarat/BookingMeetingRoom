from pulp import *
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor

groups = [
    {
        "id": "A",
        "main_start": 10, "main_end": 12,
        "alt_start": 13, "alt_end": 15,
        "priority": 5, "order": 1, "size": 4
    },
    {
        "id": "B",
        "main_start": 11, "main_end": 13,
        "alt_start": 15, "alt_end": 17,
        "priority": 3, "order": 2, "size": 2
    },
    {
        "id": "C",
        "main_start": 13, "main_end": 14,
        "alt_start": 15, "alt_end": 17,
        "priority": 4, "order": 3, "size": 3
    },
    {
        "id": "D",
        "main_start": 11, "main_end": 14,
        "alt_start": 15, "alt_end": 17,
        "priority": 3, "order": 4, "size": 3
    }
]

rooms = [
    {"id": "R1", "capacity": 4},
    {"id": "R2", "capacity": 6}
]

#เทรนAI
def generate_training_data(num_samples=1000):
    data = []
    labels = []
    for _ in range(num_samples):
        size = random.randint(1, 6)
        priority = random.randint(1, 5)
        duration = random.randint(1, 3)
        start = random.randint(8, 17 - duration)
        end = start + duration
        hour = start
        room_capacity = random.choice([4, 6])
        
        # สมการจำลองความต้องการ (demand)
        demand = (
            0.5 * priority +                       # ยิ่ง priority สูง ยิ่งต้องการมาก
            0.2 * size +                           # กลุ่มใหญ่ = ความต้องการสูง
            0.1 * (room_capacity - size) +         # ถ้าห้องพอดี = ดี
            0.1 * (12 - abs(hour - 12))            # ใกล้เที่ยงยิ่งดี (peak time)
        ) + random.uniform(-0.5, 0.5)              # noise นิดหน่อย
        
        data.append([size, priority, start, end, room_capacity])
        labels.append(demand)
    
    return np.array(data), np.array(labels)

X_train, y_train = generate_training_data()
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

#ใช้ AI คำนวณ demand score ของแต่ละ group
demand_scores = {}
for g in groups:
    for r in rooms:
        for slot in ["main", "alt"]:
            start = g[f"{slot}_start"]
            end = g[f"{slot}_end"]
            features = np.array([[g["size"], g["priority"], start, end, r["capacity"]]])
            demand = rf_model.predict(features)[0]
            demand_scores[(g["id"], r["id"], slot)] = demand


model = LpProblem("Room_Scheduling", LpMaximize)

# สร้างตัวแปร: x[group][room][slot] → ใช้ห้องนั้นในช่วงเวลานั้นหรือไม่
x = {}
for g in groups:
    for r in rooms:
        for slot in ["main", "alt"]:
            key = (g["id"], r["id"], slot)
            x[key] = LpVariable(f"x_{g['id']}_{r['id']}_{slot}", cat=LpBinary)

# เป้าหมาย: maximize ความสำคัญรวม
model += lpSum([g["priority"] * x[(g["id"], r["id"], slot)]
                for g in groups for r in rooms for slot in ["main", "alt"]])

# เงื่อนไข: กลุ่มหนึ่งเลือกได้แค่ช่วงเดียวและห้องเดียว
for g in groups:
    model += lpSum([x[(g["id"], r["id"], slot)]
                    for r in rooms for slot in ["main", "alt"]]) <= 1

# เงื่อนไข: ขนาดห้องต้องพอ
for g in groups:
    for r in rooms:
        for slot in ["main", "alt"]:
            if g["size"] > r["capacity"]:
                model += x[(g["id"], r["id"], slot)] == 0

# เงื่อนไข: ห้ามเวลาชนกันในห้องเดียวกัน
for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        g1, g2 = groups[i], groups[j]
        for r in rooms:
            for s1 in ["main", "alt"]:
                for s2 in ["main", "alt"]:
                    t1_start = g1[f"{s1}_start"]
                    t1_end = g1[f"{s1}_end"]
                    t2_start = g2[f"{s2}_start"]
                    t2_end = g2[f"{s2}_end"]
                    if not (t1_end <= t2_start or t2_end <= t1_start):
                        model += x[(g1["id"], r["id"], s1)] + x[(g2["id"], r["id"], s2)] <= 1

# แก้ปัญหา
model.solve()

# แสดงผลลัพธ์
for g in groups:
    for r in rooms:
        for slot in ["main", "alt"]:
            if x[(g["id"], r["id"], slot)].value() == 1:
                start = g[f"{slot}_start"]
                end = g[f"{slot}_end"]
                score = demand_scores[(g["id"], r["id"], slot)]
                print(f"✅ กลุ่ม {g['id']} ได้ใช้ห้อง {r['id']} ช่วง [{start}–{end}] (score={score:.2f})")