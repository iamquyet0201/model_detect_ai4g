from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageFile, ImageDraw
from collections import Counter
from typing import List
import os, base64, io, asyncio, json, re, functools, time
import numpy as np
import requests

# =========================
# Cấu hình và biến môi trường
# =========================
# Đường dẫn model YOLO
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
# Kích thước ảnh tối đa để xử lý (resize nếu lớn hơn)
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1280))
# Ngưỡng tự tin tối thiểu của YOLO
PRED_CONF = float(os.environ.get("PRED_CONF", 0.30))
# Các định dạng ảnh cho phép
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}

# Giới hạn số lượng vật thể tối đa gửi lên Gemini để check (tránh spam API)
MAX_GEMINI_CHECKS = int(os.environ.get("MAX_GEMINI_CHECKS", 20))
# Bật/tắt log in ra terminal
LOG_GEMINI = bool(int(os.environ.get("LOG_GEMINI", "1")))

# API Key Gemini (Thay thế bằng key thật của bạn hoặc set biến môi trường)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or "AIzaSyChtK9Y6ZrvV4LZoPd3k36Zov8BOyDYSzY"

# Cấu hình Gemini
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash") # Khuyên dùng bản flash mới nhất cho nhanh
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Khởi tạo App FastAPI
# =========================
app = FastAPI(title="AI 4 Green - Logic: Agree=YOLO, Disagree=Gemini")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Tải Model YOLO
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Vui lòng kiểm tra đường dẫn.")

try:
    model = YOLO(MODEL_PATH)
    # Kiểm tra xem đang chạy trên CPU hay GPU để dễ debug
    device = model.device.type if hasattr(model, 'device') else 'unknown'
    print(f"✅ YOLO Model loaded successfully on: {device.upper()}")
except Exception as e:
    raise RuntimeError(f"Không thể load YOLO model: {e}")

# =========================
# Ontology (Danh sách nhãn) - Tiếng Việt
# =========================
CORE_LABELS_VI = {
    "plastic_bottle": "Chai nhựa",
    "aluminum_can": "Lon nhôm",
    "cardboard": "Bìa cứng",
    "paper_box": "Hộp giấy",
    "plastic_bag": "Túi nilon",
}

EXTENDED_LABELS_VI = {
    "plastic_bottle_cap": "Nắp chai nhựa",
    "plastic_cup": "Ly nhựa",
    "yogurt_cup": "Cốc sữa chua",
    "paper_cup": "Cốc giấy",
    "paper_bag": "Túi giấy",
    "tetra_pak": "Hộp sữa hoặc hộp nước",
    "noodle_wrapper": "Vỏ mì gói",
}

# Gộp tất cả nhãn
DISPLAY_LABELS_VI = {**CORE_LABELS_VI, **EXTENDED_LABELS_VI}
DISPLAY_LABELS_VI["unknown"] = "Vật liệu chưa rõ"

# Tập hợp các key (tiếng Anh) để kiểm tra hợp lệ
DISPLAY_LABELS = set(DISPLAY_LABELS_VI.keys())

# =========================
# Các hàm xử lý ảnh và YOLO
# =========================
def _resize_max_side(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

def _pil_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def _crop(img: Image.Image, xyxy, pad: int = 10):
    """Cắt ảnh theo bounding box với lề (padding)"""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    w, h = img.size
    return img.crop(
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad),
        )
    )

async def yolo_predict(img: Image.Image):
    """Chạy YOLO inference bất đồng bộ"""
    loop = asyncio.get_running_loop()
    def _run():
        # save=False, verbose=False để tối ưu tốc độ
        results = model.predict(source=img, conf=PRED_CONF, verbose=False, save=False)[0]
        # plot() trả về numpy array hình ảnh đã vẽ box
        return results.boxes, results.names, results.plot()
    return await loop.run_in_executor(None, _run)

# =========================
# Các hàm xử lý Gemini
# =========================
def safe_get_text_from_gemini_response(j) -> str:
    """Trích xuất text an toàn từ JSON response của Gemini"""
    try:
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""

def _extract_json(txt: str):
    """Cố gắng parse JSON từ string trả về của LLM"""
    if not txt: return None
    txt = txt.strip()
    # 1. Thử parse trực tiếp
    try: return json.loads(txt)
    except: pass
    # 2. Thử tìm chuỗi giữa { và }
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def _build_classify_payload(b64: str) -> dict:
    """Tạo payload gửi Gemini, yêu cầu trả về JSON strict"""
    prompt = (
        "You are a recycling material classifier. "
        "Classify the object in the image into exactly one of these labels:\n"
        f"{', '.join(sorted(DISPLAY_LABELS))}\n\n"
        "Return ONLY a JSON object: {\"classification\": {\"label\": \"<label>\", \"confidence\": <float>, \"reason\": \"<short reason>\"}}"
    )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64.split(",")[-1]}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1, # Giảm nhiệt độ để kết quả ổn định
            "maxOutputTokens": 256,
            "topP": 0.8
        },
    }

def _classify_with_gemini_sync(crop: Image.Image, max_retries: int = 2):
    """Gửi request lên Gemini (đồng bộ)"""
    if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY:
        return None

    # Resize ảnh crop nếu quá to để tiết kiệm băng thông
    crop = _resize_max_side(crop, max_side=512)
    
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    
    payload = _build_classify_payload(b64)

    for attempt in range(max_retries):
        try:
            r = requests.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=15, # Timeout 15s
            )
            if r.status_code != 200:
                if LOG_GEMINI: print(f"[Gemini] HTTP {r.status_code} attempt {attempt+1}")
                time.sleep(1)
                continue
            
            j = r.json()
            text = safe_get_text_from_gemini_response(j)
            
            if LOG_GEMINI: 
                print(f"[Gemini RAW] {text[:100]}...")

            obj = _extract_json(text)
            if obj and "classification" in obj:
                res = obj["classification"]
                # Chỉ chấp nhận nếu label nằm trong danh sách cho phép
                if res.get("label") in DISPLAY_LABELS:
                    return {
                        "label": res["label"],
                        "confidence": float(res.get("confidence", 0.0)),
                        "reason": res.get("reason", "")
                    }
        except Exception as e:
            if LOG_GEMINI: print(f"[Gemini Error] {e}")
            time.sleep(1)
            continue
            
    return None

async def classify_with_gemini(crop: Image.Image):
    """Wrapper bất đồng bộ cho hàm sync"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(_classify_with_gemini_sync, crop))

# =========================
# Routes API
# =========================
@app.get("/")
def root():
    return {
        "status": "Running",
        "logic": "Agree=YOLO, Disagree=Gemini",
        "labels": DISPLAY_LABELS_VI
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 1. Kiểm tra file
    if file.content_type not in ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh (JPEG, PNG, WEBP).")
    
    # 2. Đọc và resize ảnh
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = _resize_max_side(img)
    except Exception:
        raise HTTPException(status_code=400, detail="File ảnh bị lỗi.")

    # 3. Chạy YOLO
    try:
        boxes, names, plotted = await yolo_predict(img)
    except Exception:
        raise HTTPException(status_code=500, detail="Lỗi xử lý model YOLO.")

    # Nếu không detect được gì
    if boxes is None or len(boxes) == 0:
        return {
            "items": [], 
            "detections": [], 
            "image": _pil_to_base64_jpeg(img),
            "message": "Không tìm thấy vật thể."
        }

    # 4. Chuẩn bị dữ liệu để xử lý song song
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()

    dets = []
    gemini_tasks = []
    gemini_indices = [] # Lưu chỉ số của các box được gửi đi Gemini

    # Lọc và tạo task
    for i, (b, cf, c) in enumerate(zip(xyxy, confs, clss)):
        yolo_label = names.get(int(c), str(int(c)))
        
        # Lưu thông tin cơ bản
        dets.append({
            "box": b.tolist(),
            "yolo_label": yolo_label,
            "yolo_conf": float(cf),
            # Các trường sau sẽ được update sau khi Gemini chạy xong
            "final_label": yolo_label, 
            "final_conf": float(cf),
            "source": "yolo_pending",
            "reason": ""
        })

        # Logic gửi Gemini: Nếu API Key có thì gửi (giới hạn số lượng)
        if i < MAX_GEMINI_CHECKS and GEMINI_API_KEY and "YOUR_KEY" not in GEMINI_API_KEY:
            crop = _crop(img, b)
            gemini_tasks.append(classify_with_gemini(crop))
            gemini_indices.append(i) # Đánh dấu detection thứ i đang chờ Gemini

    # 5. Chạy Gemini song song
    if gemini_tasks:
        gemini_results = await asyncio.gather(*gemini_tasks)
        
        # 6. Hợp nhất kết quả (Ensemble Logic)
        for idx_in_dets, gres in zip(gemini_indices, gemini_results):
            d = dets[idx_in_dets]
            yolo_lbl = d["yolo_label"]
            yolo_cnf = d["yolo_conf"]

            # Mặc định (nếu Gemini lỗi hoặc null) thì giữ nguyên YOLO
            final_lbl = yolo_lbl
            final_cnf = yolo_cnf
            src = "yolo_only"
            reason = ""

            if gres: # Gemini trả về kết quả thành công
                gem_lbl = gres["label"]
                gem_cnf = gres["confidence"]
                reason = gres["reason"]

                # ====================================================
                # LOGIC CHÍNH BẠN YÊU CẦU NẰM Ở ĐÂY
                # ====================================================
                if gem_lbl == yolo_lbl:
                    # TRƯỜNG HỢP 1: Giống nhau -> Lấy kết quả của MODEL CỦA BẠN (YOLO)
                    final_lbl = yolo_lbl
                    final_cnf = yolo_cnf # Giữ nguyên độ tin cậy của YOLO
                    src = "yolo_verified" # Đánh dấu là YOLO đã được verify
                else:
                    # TRƯỜNG HỢP 2: Khác nhau -> Auto lấy kết quả của API (Gemini)
                    final_lbl = gem_lbl
                    final_cnf = gem_cnf # Lấy độ tin cậy của Gemini
                    src = "gemini_correction" # Đánh dấu là Gemini sửa lỗi

                # Lưu thêm thông tin debug
                d["gemini_label"] = gem_lbl
                d["gemini_conf"] = gem_cnf

            # Cập nhật lại detection
            # Đảm bảo label cuối cùng có trong danh sách hiển thị, nếu không fallback về unknown
            if final_lbl not in DISPLAY_LABELS:
                final_lbl = "unknown"

            d["final_label"] = final_lbl
            d["final_conf"] = final_cnf
            d["source"] = src
            d["reason"] = reason
            d["label_vi"] = DISPLAY_LABELS_VI.get(final_lbl, final_lbl)

    else:
        # Nếu không chạy Gemini, chỉ cần update label tiếng Việt cho YOLO
        for d in dets:
            d["source"] = "yolo_only"
            if d["final_label"] not in DISPLAY_LABELS:
                d["final_label"] = "unknown"
            d["label_vi"] = DISPLAY_LABELS_VI.get(d["final_label"], d["final_label"])

    # 7. Tổng hợp thống kê
    counts = Counter([d["final_label"] for d in dets])
    items = [
        {"name": k, "label": DISPLAY_LABELS_VI.get(k, k), "quantity": v}
        for k, v in counts.items()
    ]

    # 8. Vẽ lại ảnh (Debug visual)
    # Convert numpy/plotted sang PIL để vẽ đè lên
    if isinstance(plotted, np.ndarray):
        # YOLO trả BGR -> RGB
        if plotted.ndim == 3 and plotted.shape[2] == 3:
            img_plt = Image.fromarray(plotted[:, :, ::-1])
        else:
            img_plt = Image.fromarray(plotted)
    else:
        img_plt = plotted

    draw = ImageDraw.Draw(img_plt)
    for d in dets:
        # Chọn màu khung: Xanh lá nếu đồng thuận, Đỏ cam nếu Gemini sửa
        color = "#00FF00" if "yolo" in d["source"] else "#FF4500"
        
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Text hiển thị: Tên tiếng Việt + Nguồn
        caption = f"{d['label_vi']} ({d['source']})"
        draw.text((x1, max(0, y1 - 15)), caption, fill=color)

    return {
        "items": items,
        "detections": dets,
        "image": _pil_to_base64_jpeg(img_plt)
    }
