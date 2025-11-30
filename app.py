import os
import io
import json
import base64
import asyncio
import time
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

# --- [QUAN TRỌNG] API KEY CỦA BẠN ---
GEMINI_API_KEY = "AIzaSyBwuuO13U2Oipb37hRaCFKavAREc-Ghjkg"  # <--- THAY API KEY VÀO ĐÂY

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

# Mapping Label
LABEL_MAP_VI: Dict[str, str] = {
    "plastic_bottle": "Chai nhựa", "aluminum_can": "Lon nhôm", "cardboard": "Bìa cứng",
    "paper_box": "Hộp giấy", "plastic_bag": "Túi nilon", "plastic_bottle_cap": "Nắp chai nhựa",
    "plastic_cup": "Ly nhựa", "yogurt_cup": "Cốc sữa chua", "paper_cup": "Cốc giấy",
    "paper_bag": "Túi giấy", "tetra_pak": "Hộp sữa/nước", "noodle_wrapper": "Vỏ mì gói",
    "unknown": "Vật liệu chưa rõ"
}
VALID_LABELS: Set[str] = set(LABEL_MAP_VI.keys())

# ==========================================
# 2. FASTAPI INIT
# ==========================================

app = FastAPI(title="AI 4 Green API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Load YOLO
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
# 3. UTILS
# ==========================================

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def safe_parse_json(text: str, default):
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except:
        return default

# ==========================================
# 4. GEMINI FUNCTIONS
# ==========================================

async def check_safety(img: Image.Image) -> Tuple[bool, str]:
    if not GEMINI_MODEL: return True, "Skipped"
    try:
        prompt = 'Check for DANGEROUS items (weapons, fire, toxic, drugs). Return JSON: {"is_safe": true, "reason": "..."}'
        res = await asyncio.to_thread(GEMINI_MODEL.generate_content, [prompt, img])
        parsed = safe_parse_json(res.text, {"is_safe": True, "reason": ""})
        return parsed.get("is_safe", True), parsed.get("reason", "")
    except:
        return True, "Error"

async def scan_gemini_labels(img: Image.Image) -> Set[str]:
    if not GEMINI_MODEL: return set()
    try:
        allowed = ", ".join(sorted(VALID_LABELS))
        prompt = f'Identify recyclables. Choose from: {allowed}. Return JSON list: ["plastic_bottle", ...]'
        res = await asyncio.to_thread(GEMINI_MODEL.generate_content, [prompt, img])
        parsed = safe_parse_json(res.text, [])
        return {s for s in parsed if s in VALID_LABELS} if isinstance(parsed, list) else set()
    except:
        return set()

# ==========================================
# 5. PREDICT API (LOGIC CHÍNH)
# ==========================================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    
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
        return JSONResponse({
            "items": [], 
            "error": "SAFETY_BLOCKED", 
            "message": f"⚠️ CẢNH BÁO: {reason}"
        }, status_code=400)

    # 3. YOLO Detect (Có NMS để lọc trùng)
    yolo_counts = Counter()
    detections = []
    
    if model:
        # Thêm agnostic_nms=True để lọc trùng lặp tốt hơn
        results = model.predict(img, conf=PRED_CONF, iou=0.5, agnostic_nms=True, verbose=False)[0]
        for box in results.boxes:
            c = int(box.cls[0])
            lbl = results.names[c]
            yolo_counts[lbl] += 1
            detections.append({"box": box.xyxy[0].tolist(), "label": lbl})

    # 4. Gemini Scan
    gemini_labels = await scan_gemini_labels(img)
    
    # 5. LOGIC KẾT HỢP (Final Merge)
    final_items = []
    processed_labels = set()

    # A. Xử lý những cái YOLO nhìn thấy
    for lbl, count in yolo_counts.items():
        name_vi = LABEL_MAP_VI.get(lbl, lbl)
        processed_labels.add(lbl)
        
        item = {
            "name": lbl,
            "label": name_vi,
            "quantity": count,
            "manual_input_required": False
        }

        if gemini_labels:
            if lbl in gemini_labels:
                # Cả 2 cùng thấy -> Tin số lượng YOLO
                item["note"] = "✅ Verified"
            else:
                # YOLO thấy, Gemini không thấy -> Vẫn hiện, nhưng đánh dấu để user check
                # (Không bắt nhập tay, cứ để số lượng YOLO gợi ý)
                item["note"] = "⚠️ Kiểm tra lại"
                # Nếu muốn bắt nhập tay thì uncomment dòng dưới:
                # item["manual_input_required"] = True 
                # item["quantity"] = 0
        
        final_items.append(item)

    # B. Xử lý những cái Gemini thấy thêm (YOLO bỏ sót)
    for lbl in gemini_labels:
        if lbl not in processed_labels:
            name_vi = LABEL_MAP_VI.get(lbl, lbl)
            final_items.append({
                "name": lbl,
                "label": name_vi,
                "quantity": 0, # Để 0 để báo hiệu chưa đếm được
                "manual_input_required": True, # Bắt buộc nhập tay
                "note": "✨ Gemini phát hiện thêm"
            })

    # 6. Vẽ hình
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    for det in detections:
        draw.rectangle(det["box"], outline="green", width=3)

    print(f"Done in {time.time() - start:.2f}s")

    return {
        "items": final_items,
        "image": pil_to_base64(draw_img)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
