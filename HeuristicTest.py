from pulp import * # ยังคง import ไว้เผื่อเปรียบเทียบ แต่เราจะไม่เรียกใช้
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
from datetime import datetime

# -----------------------------
# ข้อมูลกิจกรรม และ Priority
# -----------------------------
activities = {
    "กิจกรรม : ประชุม": 5,
    "กิจกรรม : พรีเซนต์งาน": 4,
    "กิจกรรม : กิจกรรมชมรม": 3,
    "กิจกรรม : ทำงาน/ทำการบ้าน": 2,
    "กิจกรรม : พักผ่อน": 1
}

# -----------------------------
# ข้อมูลห้อง
# -----------------------------
rooms = [
    {"id": "COC air 1", "capacity": 4},
    {"id": "COC air 2", "capacity": 6},
    {"id": "COC common", "capacity": 8}
]

# บันทึกข้อมูลลง ไฟล์ .txt
def save_groups(groups):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"Booking_{today}.txt"
    # เขียนทับไฟล์ใหม่ทุกครั้งที่มีการบันทึก
    with open(filename, "w", encoding="utf-8") as f:
        for group in groups:
            json_line = json.dumps(group, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"✅ บันทึกข้อมูล {len(groups)} กลุ่มเรียบร้อยแล้ว")
    print("-----------------------------")


# โหลดข้อมูลจาก ไฟล์ .txt
def load_groups():
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"Booking_{today}.txt"
    groups = []
    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    group = json.loads(line)
                    groups.append(group)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON ผิดรูปแบบในบรรทัด: {line}")
    except FileNotFoundError:
        print("ยังไม่มีไฟล์ข้อมูลการจองของวันนี้")
    return groups


# คำนวณระยะเวลา
def cal_duration_main(main_start, main_end):
    duration_main = main_end - main_start
    return duration_main

def cal_duration_alt(alt_start, alt_end):
    duration_alt = alt_end - alt_start
    return duration_alt

# รับข้อมูล
def input_group(order):
    print("=== ข้อมูลกลุ่ม ===")
    id = input("ชื่อกลุ่ม (ภาษาอังกฤษ): ")

    print("เลือกกิจกรรม:")
    for i, act in enumerate(activities.keys(), 1):
        print(f"{i}. {act}")
    
    activity_name = ""
    priority = 0
    while True:
        try:
            choice = int(input("เลือกกิจกรรม (1-5): "))
        except ValueError:
            print("กรุณาใส่ตัวเลขจากรายการ")
            continue
        if 1 <= choice <= len(activities):
            activity_name = list(activities.keys())[choice-1]
            priority = activities[activity_name]
            print(f"✅ คุณเลือก {activity_name}")
            break
        else:
            print("กรุณาเลือกตัวเลขจากรายการ")

    while True:
        try:
            main_start = float(input("เวลาเริ่ม (หลัก): "))
            main_end = float(input("เวลาสิ้นสุด (หลัก): "))
            if 8 <= main_start < main_end <= 18:
                break
            else:
                print("กรุณาใส่เวลาในช่วง 8.00 - 18.00 น.")
        except ValueError:
            print("กรุณาใส่ตัวเลขจากรายการ")

    while True:
        try:
            alt_start = float(input("เวลาเริ่ม (สำรอง): "))
            alt_end = float(input("เวลาสิ้นสุด (สำรอง): "))
            if 8 <= alt_start < alt_end <= 18:
                break
            else:
                print("กรุณาใส่เวลาในช่วง 8.00 - 18.00 น.")
        except ValueError:
            print("กรุณาใส่ตัวเลขจากรายการ")

    
    duration_main = cal_duration_main(main_start, main_end)
    duration_alt = cal_duration_alt(alt_start, alt_end)

    size = int(input("จำนวนผู้เข้าร่วม: "))
    return {
        "order": order,
        "id": id,
        "activity": activity_name,
        "main_start": main_start,
        "main_end": main_end,
        "alt_start": alt_start,
        "alt_end": alt_end,
        "priority": priority,
        "size": size,
        "duration_main": duration_main,
        "duration_alt": duration_alt
    }

# -----------------------------
# [Heuristic] ฟังก์ชันคำนวณคะแนน
# -----------------------------
def calculate_heuristic_score(group, room, slot):
    """คำนวณคะแนนความน่าสนใจของการจับคู่ (group, room, slot)"""
    # กำหนดค่าน้ำหนัก
    w1_priority = 10      # น้ำหนักสำหรับ Priority
    w2_main_slot = 5        # น้ำหนักโบนัสสำหรับเวลาหลัก
    w3_wasted_space = 0.5   # น้ำหนักติดลบสำหรับพื้นที่ที่เสียไป
    w4_order = 1          # น้ำหนักสำหรับลำดับการจอง (ถ้าต้องการ)

    priority = group["priority"]
    bonus_main = 1 if slot == "main" else 0
    wasted_space = max(0, room["capacity"] - group["size"])

    # คำนวณคะแนน
    score = (w1_priority * priority) + \
            (w2_main_slot * bonus_main) - \
            (w3_wasted_space * wasted_space) + \
            (w4_order * (1 / group["order"]))  # ยิ่งจองก่อน ยิ่งได้คะแนนมาก
    
    return score

# -----------------------------
# [Heuristic] ฟังก์ชันจัดตารางด้วย Greedy Algorithm
# -----------------------------
def schedule_with_heuristic(groups, rooms):
    """จัดสรรห้องโดยใช้ Greedy Algorithm ตามคะแนน Heuristic"""
    possible_assignments = []

    # 1. สร้างและคำนวณคะแนนทุกการจับคู่ที่เป็นไปได้
    for g in groups:
        for r in rooms:
            if g["size"] <= r["capacity"]: # เงื่อนไข: ขนาดกลุ่มต้องไม่เกินความจุ
                for slot in ["main", "alt"]:
                    score = calculate_heuristic_score(g, r, slot)
                    possible_assignments.append({
                        "group": g,
                        "room": r,
                        "slot": slot,
                        "score": score,
                        "start": g[f"{slot}_start"],
                        "end": g[f"{slot}_end"]
                    })

    # 2. เรียงลำดับการจับคู่ตามคะแนนจากมากไปน้อย
    sorted_assignments = sorted(possible_assignments, key=lambda x: x["score"], reverse=True)

    final_assignments = []
    assigned_groups = set()
    booked_slots = {} # dict เก็บ track เวลาที่ถูกจองของแต่ละห้อง

    # 3. วนลูปเพื่อจัดสรร (Greedy selection)
    for assignment in sorted_assignments:
        group_id = assignment["group"]["id"]
        room_id = assignment["room"]["id"]
        start_time = assignment["start"]
        end_time = assignment["end"]

        # ตรวจสอบว่ากลุ่มนี้ถูกจัดสรรไปแล้วหรือยัง
        if group_id in assigned_groups:
            continue

        # ตรวจสอบว่าช่วงเวลานี้ในห้องนี้ว่างหรือไม่
        is_conflict = False
        if room_id in booked_slots:
            for booked_start, booked_end in booked_slots[room_id]:
                if not (end_time <= booked_start or start_time >= booked_end):
                    is_conflict = True
                    break
        
        if is_conflict:
            continue

        # ถ้าไม่มีปัญหา ก็ทำการจัดสรร
        final_assignments.append(assignment)
        assigned_groups.add(group_id)
        
        if room_id not in booked_slots:
            booked_slots[room_id] = []
        booked_slots[room_id].append((start_time, end_time))

    return final_assignments
### --- จบส่วนที่เพิ่ม/แก้ไข --- ###


# (ส่วนของ AI ไม่มีการเปลี่ยนแปลง สามารถคงไว้เหมือนเดิม)
# -----------------------------
# [AI] พยากรณ์ความต้องการใช้ห้องล่วงหน้าแบบรายชั่วโมง [/AI]
# -----------------------------
def forecast_hourly_demand(groups, rooms, rf_model):
    time_demand = {hour: [] for hour in range(8, 18)}

    for group in groups:
        priority = group["priority"]
        alt_start = group["alt_start"]
        alt_end = group["alt_end"]
        duration_alt = group.get("duration_alt", alt_end - alt_start)
        size = group["size"]

        for hour in range(8, 18):
            duration_main = 1
            main_start = hour
            main_end = hour + duration_main
            if main_end > 17:
                continue

            for room in rooms:
                room_capacity = room["capacity"]
                features = np.array([[size, priority, main_start, main_end, alt_start, alt_end, duration_main, duration_alt, room_capacity, hour]])
                demand = rf_model.predict(features)[0]
                time_demand[hour].append(demand)

    # คำนวณค่าเฉลี่ย
    avg_time_demand = {}
    for hour in time_demand:
        values = time_demand[hour]
        avg = sum(values) / len(values) if values else 0
        avg_time_demand[hour] = avg
        
    return avg_time_demand

def generate_training_data(num_samples=1000):
    data, labels = [], []
    for _ in range(num_samples):
        activity = random.choice(list(activities.keys()))
        priority = activities[activity]
        duration_main = random.randint(1, 3)
        main_start = random.randint(8, 18 - duration_main)
        main_end = main_start + duration_main
        duration_alt = random.randint(1, 3)
        alt_start = random.randint(8, 17 - duration_alt)
        alt_end = alt_start + duration_alt
        size = random.randint(1, 10)
        hour = main_start
        room = random.choice(rooms)
        room_capacity = room["capacity"]
        demand = (0.5 * priority + 0.2 * size + 0.1 * (room_capacity - size) + 0.1 * (12 - abs(hour - 12))) + random.uniform(-0.5, 0.5)
        data.append([size, priority, main_start, main_end, alt_start, alt_end, duration_main, duration_alt, room_capacity, hour])
        labels.append(demand)
    return np.array(data), np.array(labels)

# -----------------------------
# Main program
# -----------------------------
print("🚀 เริ่มต้นโปรแกรม AI จัดการตารางการใช้ห้อง COC 🚀")

# โหลดข้อมูลการจอง (ถ้ามี)
groups = load_groups()

# [AI] เทรนโมเดล RandomForest 
X_train, y_train = generate_training_data()
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)


# จัดสรรตารางด้วย Heuristic ทันทีหลังโหลดข้อมูล
assignments = schedule_with_heuristic(groups, rooms)

print(f"\nมีข้อมูลกลุ่มที่บันทึกไว้ {len(groups)} กลุ่ม, จัดสรรได้ {len(assignments)} กลุ่ม")
print("="*20)
print("เลือกเมนู")
print("1. 🔎 ดูตารางการจองที่จัดสรรแล้ว")
print("2. 📝 เพิ่มการจอง")
print("3. 📊 วิเคราะห์แนวโน้มการใช้งาน")
print("4. 🚪 ออก")
print("="*20)

choice = ""
while choice != "4":
    choice = input("เลือกเมนู (1-4): ")

    if choice == "1":
        ### --- ส่วนที่เพิ่ม/แก้ไข --- ###
        if not assignments:
            print("❌ ยังไม่มีการจองที่จัดสรรได้")
        else:
            print("\n=== 📅 ตารางการจอง ===")
            # เรียงตามเวลาเริ่มเพื่อให้อ่านง่าย
            sorted_display = sorted(assignments, key=lambda x: x['start'])
            for assign in sorted_display:
                g = assign["group"]
                r = assign["room"]
                start = assign["start"]
                end = assign["end"]
                score = assign["score"]
                print(f"🔹 {start:05.2f} - {end:05.2f} | ห้อง: {r['id']:<12} | กลุ่ม: {g['id']:<10} (คะแนน: {score:.2f})")
        print("="*20)
        ### --- จบส่วนที่เพิ่ม/แก้ไข --- ###

    elif choice == "2":
        order = len(groups) + 1
        print(f"📩 เพิ่มการจองกลุ่มที่ {order}")
        new_group = input_group(order)
        groups.append(new_group)
        save_groups(groups) # บันทึกข้อมูลทั้งหมดลงไฟล์

        # โหลดข้อมูลใหม่และจัดตารางใหม่
        groups = load_groups() 
        assignments = schedule_with_heuristic(groups, rooms)
        print(f"🔄 จัดตารางใหม่เรียบร้อย! จัดสรรได้ {len(assignments)} กลุ่ม")
        print("="*20)


    elif choice == "3":
        if not groups:
            print("❌ ยังไม่มีข้อมูลการจองให้วิเคราะห์")
        else: 
            print("\n=== 📊 วิเคราะห์แนวโน้มการใช้งาน ===")
            avg_time_demand = forecast_hourly_demand(groups, rooms, rf_model)
            for hour, demand in avg_time_demand.items():
                bar = '█' * int(demand * 2)
                print(f"{hour:02d}.00 - {hour+1:02d}.00 | Demand: {demand:4.2f} | {bar}")
            
            min_hour = min(avg_time_demand, key=avg_time_demand.get)
            max_hour = max(avg_time_demand, key=avg_time_demand.get)
            print("\n🔹 สรุปช่วงเวลา:")
            print(f"   ⬇️ ใช้งานน้อยที่สุด: {min_hour:02d}.00 - {min_hour+1:02d}.00")
            print(f"   ⬆️ ใช้งานมากที่สุด: {max_hour:02d}.00 - {max_hour+1:02d}.00")
        print("="*20)


    elif choice == "4":
        print("ขอบคุณที่ใช้บริการ 🙏")
        break

    else:
        print("❌ กรุณาเลือกเมนู 1-4")