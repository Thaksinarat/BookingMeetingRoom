# from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary

# # ข้อมูลกลุ่มที่ต้องการจอง
# groups = [
#     {"id": "A", "start": 10, "end": 12, "priority": 3, "order": 1},
#     {"id": "B", "start": 11, "end": 13, "priority": 5, "order": 2},
#     {"id": "C", "start": 14, "end": 16, "priority": 4, "order": 3}
# ]

# # สร้างตัวแปร decision: ให้จองหรือไม่จอง
# x = {g["id"]: LpVariable(f"x_{g['id']}", cat=LpBinary) for g in groups}

# # สร้าง model
# model = LpProblem("Room_Scheduling", LpMaximize)

# # เป้าหมาย: maximize ความสำคัญรวมของกิจกรรม
# model += lpSum([g["priority"] * x[g["id"]] for g in groups])

# # เงื่อนไข: ห้ามเวลาชนกัน
# for i in range(len(groups)):
#     for j in range(i + 1, len(groups)):
#         g1, g2 = groups[i], groups[j]
#         if not (g1["end"] <= g2["start"] or g2["end"] <= g1["start"]):
#             # ถ้าเวลาชนกัน → เลือกได้แค่กลุ่มเดียว
#             model += x[g1["id"]] + x[g2["id"]] <= 1

# # แก้ปัญหา
# model.solve()

# # แสดงผลลัพธ์
# for g in groups:
#     if x[g["id"]].value() == 1:
#         print(f"✅ กลุ่ม {g['id']} ได้ใช้ห้องช่วง {g['start']}–{g['end']}")
#     else:
#         print(f"❌ กลุ่ม {g['id']} ไม่ได้ใช้ห้อง")


# AI จัดการ optimization
from pulp import *

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
                print(f"✅ กลุ่ม {g['id']} ได้ใช้ห้อง {r['id']} ช่วง {start}–{end}")

