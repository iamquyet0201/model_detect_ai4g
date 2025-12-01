import os
import io
import json
import base64
import asyncio
import time
import re
from collections import Counter
from typing import List, Dict, Set, Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from ultralytics import YOLO
import google.generativeai as genai

# ==========================================
# 1. CONFIG
# ==========================================

# --- [QUAN TRỌNG] API KEY ---
GEMINI_API_KEY = "AIzaSyDsvSCONTmJlcY4QcTEbDIYxXRJ6Zpgcfo"  # <--- THAY KEY CỦA BẠN

MODEL_PATH = "best.pt"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
PRED_CONF = 0.25

# Setup Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"✅ Gemini model loaded: {GEMINI_MODEL_NAME}")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Gemini: {e}")
    GEMINI_MODEL = None

# Mapping Label (Key chuẩn -> Tên hiển thị)
LABEL_MAP_VI: Dict[str, str] = {
    "plastic_bottle": "Chai nhựa", "aluminum_can": "Lon nhôm", "cardboard": "Bìa cứng",
    "paper_box": "Hộp giấy", "plastic_bag": "Túi nilon", "plastic_bottle_cap": "Nắp chai nhựa",
    "plastic_cup": "Ly nhựa", "yogurt_cup": "Cốc sữa chua", "paper_cup": "Cốc giấy",
    "paper_bag": "Túi giấy", "tetra_pak": "Hộp sữa/nước", "noodle_wrapper": "Vỏ mì gói",
    "unknown": "Vật liệu chưa rõ"
}
VALID_LABELS = list(LABEL_MAP_VI.keys())

# ==========================================
# 2. FASTAPI INIT
# ==========================================

app = FastAPI(title="AI 4 Green API - Robust Gemini")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded.")
    else:
        model = None
        print(f"⚠️ Không tìm thấy file {MODEL_PATH}")
except Exception as e:
    print("❌ Lỗi load YOLO:", e)
    model = None

# ==========================================
# 3. UTILS (HÀM LÀM SẠCH MẠNH MẼ)
# ==========================================

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def normalize_key(text: str) -> str:
    """Chuẩn hóa mọi kiểu viết của Gemini về key chuẩn"""
    # Xóa dấu, ký tự lạ, chuyển về chữ thường
    text = text.lower().strip()
    text = text.replace(" ", "_").replace("-", "_")
    return text

def fuzzy_match_label(raw_label: str) -> str:
    """Cố gắng tìm key chuẩn từ text lộn xộn của Gemini"""
    norm = normalize_key(raw_label)
    
    # 1. Check chính xác
    if norm in LABEL_MAP_VI:
        return norm
        
    # 2. Check chứa trong (VD: "large_plastic_bottle" -> "plastic_bottle")
    for valid_key in VALID_LABELS:
        if valid_key in norm or norm in valid_key:
            return valid_key
            
    return None

def robust_json_parse(text: str):
    """Cố gắng parse JSON từ mọi định dạng rác"""
    try:
        # Cách 1: Parse trực tiếp
        return json.loads(text)
    except:
        try:
            # Cách 2: Tìm nội dung trong ```json ... ```
            match = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Cách 3: Tìm mảng [...] hoặc object {...} đầu tiên
            match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            return None
    return None

# ==========================================
# 4. GEMINI FUNCTIONS
# ==========================================

async def check_safety(img: Image.Image) -> Tuple[bool, str]:
    if not GEMINI_MODEL: return True, "Skipped"
    try:
        prompt = 'Check for DANGEROUS items (weapons, fire, toxic, drugs). Return JSON: {"is_safe": true, "reason": "..."}'
        res = await asyncio.to_thread(GEMINI_MODEL.generate_content, [prompt, img])
        parsed = robust_json_parse(res.text)
        if parsed:
            return parsed.get("is_safe", True), parsed.get("reason", "")
        return True, ""
    except:
        return True, "Error"

async def scan_gemini_labels(img: Image.Image) -> List[str]:
    if not GEMINI_MODEL: return []
    try:
        # Prompt rõ ràng hơn, yêu cầu tiếng Anh chuẩn
        allowed_str = ", ".join(VALID_LABELS)
        prompt = (
            f"Identify recyclables in this image. Only select from this list: [{allowed_str}]. "
            "Return a JSON List of strings. Example: [\"plastic_bottle\", \"cardboard\"]. "
            "If nothing found, return []."
        )
        
        res = await asyncio.to_thread(GEMINI_MODEL.generate_content, [prompt, img])
        print(f"🔹 Gemini Raw Response: {res.text}") # Debug log
        
        parsed = robust_json_parse(res.text)
        
        valid_results = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    matched_key = fuzzy_match_label(item)
                    if matched_key:
                        valid_results.append(matched_key)
                        
        return list(set(valid_results)) # Unique
    except Exception as e:
        print(f"❌ Gemini Scan Error: {e}")
        return []

# ==========================================
# 5. PREDICT API
# ==========================================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    print(f"\n--- Request: {file.filename} ---")
    
    # 1. Load Image
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.thumbnail((1024, 1024)) 
    except:
        return JSONResponse({"error": "Lỗi file ảnh"}, status_code=400)

    # 2. Safety Check
    is_safe, reason = await check_safety(img)
    if not is_safe:
        print(f"🛑 Blocked: {reason}")
        return JSONResponse({
            "items": [], 
            "error": "SAFETY_BLOCKED", 
            "message": f"⚠️ CẢNH BÁO: {reason}"
        }, status_code=400)

    # 3. YOLO Detect
    yolo_counts = Counter()
    detections = []
    
    if model:
        results = model.predict(img, conf=PRED_CONF, iou=0.5, agnostic_nms=True, verbose=False)[0]
        for box in results.boxes:
            c = int(box.cls[0])
            lbl = results.names[c]
            yolo_counts[lbl] += 1
            detections.append({"box": box.xyxy[0].tolist(), "label": lbl})
    
    print(f"👁️ YOLO thấy: {dict(yolo_counts)}")

    # 4. Gemini Scan
    print("🧠 Gemini đang quét...")
    gemini_labels = await scan_gemini_labels(img)
    print(f"🧠 Gemini thấy: {gemini_labels}")
    
    # 5. LOGIC HỢP NHẤT (CÓ FALLBACK)
    final_items = []
    
    # Nếu Gemini KHÔNG thấy gì (hoặc lỗi), dùng YOLO làm phương án dự phòng
    if not gemini_labels:
        print("⚠️ Gemini trả về rỗng -> Dùng kết quả YOLO (Fallback)")
        for lbl, count in yolo_counts.items():
            name_vi = LABEL_MAP_VI.get(lbl, lbl)
            final_items.append({
                "name": lbl,
                "label": name_vi,
                "quantity": count,
                "manual_input_required": False, # Tin YOLO
                "note": "Backup (YOLO)"
            })
    else:
        # Nếu Gemini có kết quả, chạy logic ưu tiên Gemini
        processed_labels = set()
        
        # Vòng 1: Duyệt theo Gemini (Chính)
        for gem_lbl in gemini_labels:
            name_vi = LABEL_MAP_VI.get(gem_lbl, gem_lbl)
            processed_labels.add(gem_lbl)
            
            yolo_qty = yolo_counts.get(gem_lbl, 0)
            
            if yolo_qty > 0:
                # Trùng -> Lấy số lượng YOLO
                final_items.append({
                    "name": gem_lbl,
                    "label": name_vi,
                    "quantity": yolo_qty,
                    "manual_input_required": False,
                    "note": "✅ Verified"
                })
            else:
                # Lệch -> Lấy Gemini, nhập tay
                final_items.append({
                    "name": gem_lbl,
                    "label": name_vi,
                    "quantity": 0,
                    "manual_input_required": True,
                    "note": "⚠️ Cần nhập số"
                })
        
        # Vòng 2: Vớt vát YOLO (Phòng khi Gemini sót)
        for yolo_lbl, count in yolo_counts.items():
            if yolo_lbl not in processed_labels:
                name_vi = LABEL_MAP_VI.get(yolo_lbl, yolo_lbl)
                final_items.append({
                    "name": yolo_lbl,
                    "label": name_vi,
                    "quantity": count,
                    "manual_input_required": True, # Cảnh báo để user check
                    "note": "❓ Chỉ YOLO thấy"
                })

    # 6. Draw & Return
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    for det in detections:
        draw.rectangle(det["box"], outline="green", width=3)

    print(f"🏁 Done: {len(final_items)} items")
    return {
        "items": final_items,
        "image": pil_to_base64(draw_img)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
