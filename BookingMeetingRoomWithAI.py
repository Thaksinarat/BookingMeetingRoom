import os
import json
import random
import re
import difflib
import numpy as np
from datetime import datetime
from pulp import *
from sklearn.ensemble import RandomForestRegressor

# --- NLP Library ---
# ใช้ PyThaiNLP แทน custom import เพื่อความเสถียร (หรือเปลี่ยนกลับเป็น newmm_tokenizer ของคุณได้)
try:
    from pythainlp.tokenize import word_tokenize
except ImportError:
    # Fallback หรือแจ้งเตือนให้ลง library
    print("⚠️ ไม่พบ PyThaiNLP: กรุณาติดตั้งโดยใช้ 'pip install pythainlp'")
    # Dummy function เพื่อกันโปรแกรมพังถ้ารันโดยไม่มี lib
    def word_tokenize(text, engine="newmm"):
        return text.split()

# -----------------------------
# ข้อมูลกิจกรรม และ Priority
# -----------------------------
ACTIVITY_CONFIG = [
    {
        "category": "Meeting/Work",
        "priority": 5,
        "keywords": ["ประชุม", "meet", "conf", "discuss", "คุยงาน", "บรีฟ"]
    },
    {
        "category": "Presentation",
        "priority": 4,
        "keywords": ["พรีเซน", "เสนอ", "present", "pitch", "demo", "ขายงาน"]
    },
    {
        "category": "Study/Club",
        "priority": 3,
        "keywords": ["เรียน", "สอบ", "ติว", "ชมรม", "class", "exam", "quiz", "club", "กิจกรรม"]
    },
    {
        "category": "Group Work",
        "priority": 2,
        "keywords": ["ทำงาน", "งานกลุ่ม", "group", "project", "homework", "assignment"]  
    },
    {
        "category": "Relax",
        "priority": 1,
        "keywords": ["นอน", "พัก", "เล่น", "game", "ดูหนัง", "กิน"]
    }
]

# -----------------------------
# ข้อมูลห้อง
# -----------------------------
rooms = [
    {"id": "COC air 1", "capacity": 8},
    {"id": "COC air 2", "capacity": 8},
    {"id": "COC common", "capacity": 12}
]

# บันทึกข้อมูลลง ไฟล์ .txt
def save_groups(groups):
    # [FIX] สร้าง Folder Data หากยังไม่มี
    os.makedirs("Data", exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"Data/Booking_{today}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for group in groups:
            json_line = json.dumps(group, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"✅ บันทึกข้อมูล {len(groups)} กลุ่มเรียบร้อยแล้ว")
    print("-----------------------------")


# โหลดข้อมูลจาก ไฟล์ .txt
def load_groups():
    # [FIX] สร้าง Folder Data หากยังไม่มี (กัน Error ตอนอ่านครั้งแรก)
    os.makedirs("Data", exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"Data/Booking_{today}.txt"
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
        print("⚠️ ยังไม่มีไฟล์ข้อมูลการจองของวันนี้ (ระบบจะสร้างใหม่เมื่อมีการจอง)")
    return groups


# ดึงกิจกรรม 
def get_activity(text):
    tokens = word_tokenize(text, engine="newmm")
    
    found_act = "General" # Default
    max_prio = 1          # Default Priority ควรเป็น 1 (ต่ำสุด) ไม่ใช่ 0
    matched_keyword = ""
    
    # วนลูปตรวจสอบแต่ละ Token
    for token in tokens:
        token_clean = token.lower().strip() # แปลงเป็นตัวเล็ก ตัดช่องว่าง
        
        if len(token_clean) < 2: continue # ข้ามคำสั้นๆ
        
        for group in ACTIVITY_CONFIG:
            is_substring = any(k in token_clean for k in group["keywords"])
            close_matches = difflib.get_close_matches(token_clean, group["keywords"], n=1, cutoff=0.75)
            
            if is_substring or close_matches:
                if group["priority"] > max_prio:
                    max_prio = group["priority"]
                    found_act = group["category"]
    
    return found_act, max_prio

# ดึงจำนวนคน
def get_size(text):
    pattern = r'(\d+)\s*(?:คน|ท่าน|ที่|ที่นั่ง|seats|participants)'
    matches = re.findall(pattern, text)

    if matches:
        people = [int(x) for x in matches]
        return people
    return []

# ดึงเวลา
def get_time(text):
    text = text.strip()
    minutes = 0.00

    # เช็คคำว่า ครึ่ง
    if "ครึ่ง" in text:
        minutes = 0.30
        text = text.replace("ครึ่ง", "").strip()
    
    if "เที่ยง" in text:
        return 12.00 + minutes
    
    if "บ่ายโมง" in text:
        return 13.00 + minutes
    
    text = text.replace(":", ".")

    nums = re.findall(r"(\d+\.?\d*)", text)

    if not nums: return 0.0 # [FIX] Return 0.0 แทน None เพื่อป้องกัน error ตอนบวกเลข
    val = float(nums[0])

    if "บ่าย" in text:
        if val <= 4:
            return (val + 12.00) + minutes
        else:
            return val + minutes
    elif "โมง" in text:
        if "เย็น" in text and val <= 6:
            return (val + 12.00) + minutes
        return val + minutes

    return val + minutes


# รับข้อมูลกลุ่ม
def input_group(order):
    print("\n--- 📝 กรอกข้อมูล ---")
    id = input("ชื่อผู้จอง/กลุ่ม: ")
    if not id: id = f"Group_{order}"

    print("ตัวอย่าง: จองห้องประชุมวิชาการ เวลา 9.00-11.30 น. จำนวน 10 คน")
    input_text = input("รายละเอียดกิจกรรม: ")
    
    # กิจกรรม
    activity_name, priority = get_activity(input_text)
    
    # [FIX] ตั้งค่าเริ่มต้น กันกรณี Regex หาไม่เจอแล้วตัวแปรไม่มีค่า
    start_time = 0.0
    end_time = 0.0
    
    # หาช่วงเวลา
    time_chunk_pattern = r'(?:เที่ยง|บ่ายโมง|บ่าย\s*\d+|(?:\d{1,2}[:.]\d{2})|(?:\d{1,2}\s*(?:โมง|น\.|นาฬิกา|ทุ่ม)))(?:\s*ครึ่ง)?'
    full_time_pattern = rf"({time_chunk_pattern})\s*(?:ถึง|-)\s*({time_chunk_pattern})"

    time_matches = re.findall(full_time_pattern, input_text)

    # วิเคราะห์ช่วงเวลา
    if time_matches:
        # เอา Match สุดท้าย หรือ Match แรกก็ได้ (ในที่นี้เอา Match แรกที่เจอ)
        raw_start, raw_end = time_matches[0]
        start_time = get_time(raw_start)
        end_time = get_time(raw_end)
    else:
        print("⚠️ ระบบตรวจจับเวลาไม่ได้ กรุณาระบุเวลาเอง (เช่น 9.00):")
        try:
            start_time = float(input("Start Time: ").replace(":", "."))
            end_time = float(input("End Time: ").replace(":", "."))
        except:
            start_time = 9.00
            end_time = 10.00

    duration_main = end_time - start_time # ระยะเวลา

    # ขนาดกลุ่ม
    size_list = get_size(input_text)
    # [FIX] แปลง List เป็น Int (เอาตัวแรก) ถ้าไม่มีให้เป็น 1
    size = size_list[0] if size_list else 1 

    # [FIX] สร้างข้อมูล Alternative Time (alt) เพื่อรองรับฟังก์ชัน AI
    # ในที่นี้ให้เท่ากับ Main Time ไปก่อน เพราะ User ไม่ได้กรอก
    alt_start = start_time
    alt_end = end_time
    duration_alt = duration_main

    return {
        "order": order,
        "id": id,
        "activity": activity_name,
        "main_start": start_time,
        "main_end": end_time,
        "priority": priority,
        "size": size,
        "duration_main": duration_main,
        # Keys สำหรับ AI ที่เพิ่มเข้ามา
        "alt_start": alt_start,
        "alt_end": alt_end,
        "duration_alt": duration_alt
    }

# -----------------------------
# คำนวณคะแนน
# -----------------------------
def calculate_heuristic_score(group, room, slot):
    w1_order = 1      
    w2_priority = 10     
    w3_main_slot = 5       
    w4_wasted_space = 0.5   

    priority = group["priority"]
    bonus_main = 1 if slot == "main" else 0
    
    # [CHECKED] group["size"] เป็น int แล้ว (แก้ใน input_group)
    wasted_space = max(0, room["capacity"] - group["size"])

    score = (w2_priority * priority) + \
            (w3_main_slot * bonus_main) - \
            (w4_wasted_space * wasted_space) + \
            (w1_order * (1 / group["order"]))
    
    return score

# -----------------------------
# จัดสรรตาราง
# -----------------------------
def schedule_with_heuristic(groups, rooms):
    possible_assignments = []

    for g in groups:
        for r in rooms:
            if g["size"] <= r["capacity"]:
                for slot in ["main"]:
                    score = calculate_heuristic_score(g, r, slot)
                    possible_assignments.append({
                        "group": g,
                        "room": r,
                        "slot": slot,
                        "score": score,
                        "start": g[f"{slot}_start"],
                        "end": g[f"{slot}_end"]
                    })

    sorted_assignments = sorted(possible_assignments, key=lambda x: x["score"], reverse=True)

    final_assignments = []
    assigned_groups = set()
    booked_slots = {} 

    for assignment in sorted_assignments:
        group_id = assignment["group"]["id"]
        room_id = assignment["room"]["id"]
        start_time = assignment["start"]
        end_time = assignment["end"]

        if group_id in assigned_groups:
            continue

        is_conflict = False
        if room_id in booked_slots:
            for booked_start, booked_end in booked_slots[room_id]:
                # Logic: Overlap Check
                if end_time > booked_start and start_time < booked_end:
                    is_conflict = True
                    break
        
        if is_conflict:
            continue
        
        final_assignments.append(assignment)
        assigned_groups.add(group_id)
        
        if room_id not in booked_slots:
            booked_slots[room_id] = []
        booked_slots[room_id].append((start_time, end_time))

    return final_assignments

# -----------------------------
# [AI] พยากรณ์ความต้องการใช้ห้อง
# -----------------------------
def forecast_hourly_demand(groups, rooms, rf_model):
    time_demand = {hour: [] for hour in range(8, 18)}

    for group in groups:
        priority = group["priority"]
        # [CHECKED] keys เหล่านี้มีแล้วจากการแก้ input_group
        alt_start = group["alt_start"] 
        alt_end = group["alt_end"]
        duration_alt = group.get("duration_alt", alt_end - alt_start)
        size = group["size"]

        for hour in range(8, 18):
            duration_main = 1
            main_start = hour
            main_end = hour + duration_main
            if main_end > 18:
                continue

            for room in rooms:
                room_capacity = room["capacity"]
                # Features ต้องตรงกับตอน Train
                features = np.array([[size, priority, main_start, main_end, alt_start, alt_end, duration_main, duration_alt, room_capacity, hour]])
                demand = rf_model.predict(features)[0]
                time_demand[hour].append(demand)

    avg_time_demand = {}
    for hour in time_demand:
        values = time_demand[hour]
        avg = sum(values) / len(values) if values else 0
        avg_time_demand[hour] = avg
        
    return avg_time_demand

def generate_training_data(num_samples=1000):
    data = []
    labels = []
    for _ in range(num_samples):
        # [FIX] ACTIVITY_CONFIG เป็น List ต้องสุ่ม dict ออกมาก่อน แล้วค่อยเรียก key
        config_item = random.choice(ACTIVITY_CONFIG)
        priority = config_item["priority"]
        
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

# โหลดข้อมูลการจอง
groups = load_groups()

# [AI] เทรนโมเดล
X_train, y_train = generate_training_data()
rf_model = RandomForestRegressor(n_estimators=10) # ลด n_estimators เพื่อความเร็วในการเทส
rf_model.fit(X_train, y_train)


# จัดสรรตาราง
assignments = schedule_with_heuristic(groups, rooms)

print(f"\nจำนวนการจองห้องวันนี้: {len(assignments)} กลุ่ม")
print("="*20)

choice = ""
while choice != "4":
    print("\nเลือกเมนู")
    print("1. 🔎 ดูตารางการจองที่จัดสรรแล้ว")
    print("2. 📝 เพิ่มการจอง")
    print("3. 📊 วิเคราะห์แนวโน้มการใช้งาน")
    print("4. 🚪 ออก")
    print("="*20)

    choice = input("เลือกเมนู (1-4): ")

    if choice == "1":
        if not assignments:
            print("❌ ยังไม่มีการจองที่จัดสรรได้")
        else:
            print("\n=== 📅 ตารางการจอง ===")
            sorted_display = sorted(assignments, key=lambda x: x['start'])
            for assign in sorted_display:
                g = assign["group"]
                r = assign["room"]
                start = assign["start"]
                end = assign["end"]
                score = assign["score"]
                print(f"🔹 {start:05.2f} - {end:05.2f} | ห้อง: {r['id']:<12} | {g['id']:<10} (Score: {score:.1f})")
        print("="*20)

    elif choice == "2":
        order = len(groups) + 1
        print(f"📩 เพิ่มการจองกลุ่มที่ {order}")
        try:
            new_group = input_group(order)
            groups.append(new_group)
            save_groups(groups)
            
            groups = load_groups() 
            assignments = schedule_with_heuristic(groups, rooms)
            print(f"✅ บันทึกและจัดตารางใหม่เรียบร้อย!")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเพิ่มข้อมูล: {e}")
        print("="*20)


    elif choice == "3":
        if not groups:
            print("❌ ยังไม่มีข้อมูลการจองให้วิเคราะห์")
        else: 
            print("\n=== 📊 วิเคราะห์แนวโน้มการใช้งาน (AI Forecast) ===")
            try:
                avg_time_demand = forecast_hourly_demand(groups, rooms, rf_model)
                
                # Visualizing
                for hour in range(8, 18):
                    demand = avg_time_demand.get(hour, 0)
                    bar_len = int(demand * 5) # scale bar
                    bar = '█' * bar_len
                    print(f"{hour:02d}.00 - {hour+1:02d}.00 | Demand: {demand:4.2f} | {bar}")
                
                if avg_time_demand:
                    min_hour = min(avg_time_demand, key=avg_time_demand.get)
                    max_hour = max(avg_time_demand, key=avg_time_demand.get)
                    print("\n🔹 สรุปช่วงเวลา:")
                    print(f"   ⬇️ ใช้งานน้อยที่สุด: {min_hour:02d}.00 - {min_hour+1:02d}.00")
                    print(f"   ⬆️ ใช้งานมากที่สุด: {max_hour:02d}.00 - {max_hour+1:02d}.00")
            except Exception as e:
                print(f"❌ AI Error: {e}")
                print("คำแนะนำ: ลองลบไฟล์ Booking เก่าในโฟลเดอร์ Data แล้วเริ่มใหม่")
        print("="*20)


    elif choice == "4":
        print("ขอบคุณที่ใช้บริการ 🙏")
        break

    else:
        print("❌ กรุณาเลือกเมนู 1-4")