from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageFile, ImageDraw
from collections import Counter
from typing import Dict, List, Literal, Optional
import os, base64, io, asyncio, json, re, functools, time
import numpy as np
import requests


# =========================
# Cấu hình & biến môi trường
# =========================
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1280))
PRED_CONF = float(os.environ.get("PRED_CONF", 0.30))
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_BG: Literal["white", "transparent"] = "white"

# Số bbox gửi Gemini
MAX_GEMINI_CHECKS = int(os.environ.get("MAX_GEMINI_CHECKS", 20))
LOG_GEMINI = bool(int(os.environ.get("LOG_GEMINI", "1")))

# API KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or "AIzaSyChtK9Y6ZrvV4LZoPd3k36Zov8BOyDYSzY"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Khởi tạo app
# =========================
app = FastAPI(title="AI 4 Green - YOLO + Gemini (Agree -> keep, Disagree -> Gemini)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =========================
# Tải YOLO
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Không thể load YOLO model: {e}")


# =========================
# Ontology
# =========================
CORE_LABELS_VI = {
    "plastic_bottle": "Chai nhựa",
    "plastic_bottle_cap": "Nắp chai nhựa",
    "paper_cup": "Cốc giấy",
    "tongue_depressor": "Que đè lưỡi",
    "cardboard": "Bìa cứng",
    "straw": "Ống hút",
}
EXTENDED_LABELS_VI = {
    "aluminum_can": "Lon nhôm",
    "steel_can": "Lon thép",
    "glass_bottle": "Chai thủy tinh",
    "glass_jar": "Hũ thủy tinh",
    "tetra_pak": "Hộp giấy Tetra Pak",
    "plastic_cup": "Ly nhựa",
    "foam_cup": "Cốc xốp",
    "plastic_bag": "Túi nilon",
    "zipper_bag": "Túi zipper",
    "snack_wrapper": "Vỏ bánh hoặc snack",
    "noodle_wrapper": "Vỏ mì gói",
    "cling_film": "Màng bọc thực phẩm",
    "aluminum_foil": "Giấy bạc nhôm",
    "paper_bag": "Túi giấy",
    "paper_plate": "Đĩa giấy",
    "plastic_container_pp": "Hộp nhựa PP",
    "plastic_container_pet": "Hộp nhựa PET",
    "detergent_bottle": "Chai nước rửa chén",
    "shampoo_bottle": "Chai dầu gội",
    "yogurt_cup": "Cốc sữa chua",
    "stirring_stick_wood": "Que khuấy gỗ",
    "paper_straw": "Ống hút giấy",
    "plastic_straw": "Ống hút nhựa",
    "battery": "Pin",
    "light_bulb": "Bóng đèn",
}
DISPLAY_LABELS_VI = {**CORE_LABELS_VI, **EXTENDED_LABELS_VI}
DISPLAY_LABELS = set(DISPLAY_LABELS_VI.keys())

LABEL_ICONS_VI = {
    "plastic_bottle": {"icon": "🧴", "color": "#4CAF50"},
    "plastic_bottle_cap": {"icon": "🔘", "color": "#4CAF50"},
    "plastic_cup": {"icon": "🥤", "color": "#4CAF50"},
    "plastic_container_pp": {"icon": "🧊", "color": "#4CAF50"},
    "plastic_container_pet": {"icon": "🧊", "color": "#4CAF50"},
    "plastic_bag": {"icon": "🛍️", "color": "#4CAF50"},
    "zipper_bag": {"icon": "🛍️", "color": "#4CAF50"},
    "plastic_straw": {"icon": "🥢", "color": "#4CAF50"},
    "foam_cup": {"icon": "🥛", "color": "#4CAF50"},
    "shampoo_bottle": {"icon": "🧴", "color": "#4CAF50"},
    "detergent_bottle": {"icon": "🧴", "color": "#4CAF50"},
    "cardboard": {"icon": "📦", "color": "#D4A017"},
    "paper_cup": {"icon": "☕", "color": "#D4A017"},
    "paper_bag": {"icon": "🛍️", "color": "#D4A017"},
    "paper_plate": {"icon": "🍽️", "color": "#D4A017"},
    "tetra_pak": {"icon": "🥫", "color": "#D4A017"},
    "paper_straw": {"icon": "🥢", "color": "#D4A017"},
    "newspaper": {"icon": "🗞️", "color": "#D4A017"},
    "magazine": {"icon": "📖", "color": "#D4A017"},
    "aluminum_can": {"icon": "🥫", "color": "#9E9E9E"},
    "steel_can": {"icon": "🥫", "color": "#9E9E9E"},
    "aluminum_foil": {"icon": "📄", "color": "#9E9E9E"},
    "glass_bottle": {"icon": "🍾", "color": "#00BCD4"},
    "glass_jar": {"icon": "🫙", "color": "#00BCD4"},
    "tongue_depressor": {"icon": "🥢", "color": "#8D6E63"},
    "stirring_stick_wood": {"icon": "🥢", "color": "#8D6E63"},
    "snack_wrapper": {"icon": "🍪", "color": "#FF9800"},
    "noodle_wrapper": {"icon": "🍜", "color": "#FF9800"},
    "cling_film": {"icon": "🎞️", "color": "#FF9800"},
    "yogurt_cup": {"icon": "🥣", "color": "#FF9800"},
    "milk_carton": {"icon": "🥛", "color": "#FF9800"},
    "juice_box": {"icon": "🧃", "color": "#FF9800"},
    "battery": {"icon": "🔋", "color": "#607D8B"},
    "light_bulb": {"icon": "💡", "color": "#FFD600"},
}


# =========================
# Helpers ảnh & YOLO
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


async def yolo_predict(img: Image.Image):
    loop = asyncio.get_running_loop()

    def _run():
        r = model.predict(source=img, conf=PRED_CONF, verbose=False, save=False)[0]
        return r.boxes, r.names, r.plot()

    return await loop.run_in_executor(None, _run)


def _crop(img: Image.Image, xyxy, pad=12):
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


# =========================
# Gemini helpers (đã làm chắc hơn)
# =========================
def safe_get_text_from_gemini_response(j) -> str:
    """
    Lấy text từ response Gemini v1, tránh lỗi khi thiếu candidates hoặc content.
    """
    if not isinstance(j, dict):
        return ""
    candidates = j.get("candidates")
    if not candidates or not isinstance(candidates, list):
        return ""
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        return ""

    content = first_candidate.get("content")
    # Có khi content là dict, có khi là list
    if isinstance(content, dict):
        parts = content.get("parts")
    elif isinstance(content, list) and content:
        parts = content[0].get("parts")
    else:
        parts = None

    if not parts or not isinstance(parts, list):
        return ""

    first_part = parts[0]
    if not isinstance(first_part, dict):
        return ""
    return first_part.get("text", "") or ""


def _extract_json(txt: str):
    """
    Parse chặt chẽ chuỗi text thành JSON.
    Xử lý cả trường hợp Gemini trả về ```json ...```.
    """
    if not txt:
        return None

    # bỏ code fence ```json ``` hoặc ```JSON ```
    txt = txt.strip()
    txt = re.sub(r"```(?:json|JSON)?", "", txt)
    txt = txt.strip("` \n\t")

    # thử parse trực tiếp
    try:
        return json.loads(txt)
    except Exception:
        pass

    # tìm block { ... } dài nhất
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        candidate = m.group(0)
        # thử parse đầy đủ
        try:
            return json.loads(candidate)
        except Exception:
            # cắt dần nếu bị rác ở cuối
            for end in range(len(candidate), 1, -1):
                try:
                    return json.loads(candidate[:end])
                except Exception:
                    continue

    # fallback tìm theo tên label trong text
    lowered = txt.lower()
    for label in DISPLAY_LABELS:
        if label.replace("_", " ") in lowered:
            return {
                "classification": {
                    "label": label,
                    "confidence": 0.6,
                    "reason": "fallback text match",
                }
            }
    return None


def _build_classify_payload(b64: str) -> dict:
    prompt = (
        "You are a JSON only classifier for recycling materials.\n"
        "Given the image, classify the object into exactly one of these labels:\n\n"
        f"{', '.join(sorted(DISPLAY_LABELS))}\n\n"
        "Return only a strict JSON object in this format:\n"
        "{\"classification\": {\"label\": \"<label>\", \"confidence\": <float>, \"reason\": \"<short reason>\"}}.\n"
        "Do not include any explanations or text outside JSON."
    )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64.split(",")[-1],
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 256,
            "topP": 0.8,
        },
    }


def _classify_with_gemini_sync(crop: Image.Image, max_retries: int = 2):
    """
    Gọi Gemini đồng bộ, luôn trả về:
      - dict {"label":..., "confidence":..., "reason":...}
      - hoặc None nếu không dùng được
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return None

    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    payload = _build_classify_payload(b64)

    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=20,
            )
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                if LOG_GEMINI:
                    print(f"[Gemini] HTTP {r.status_code} attempt {attempt+1}: {r.text[:200]}")
                time.sleep(1)
                continue

            j = r.json()
            text = safe_get_text_from_gemini_response(j)

            if LOG_GEMINI:
                if text:
                    preview = (text[:180] + "...") if len(text) > 180 else text
                    print(f"[Gemini] raw attempt {attempt+1}: {preview}")
                else:
                    print(f"[Gemini] ⚠️ Empty response attempt {attempt+1} — no candidates or empty text.")

            if not text:
                last_error = "empty_text"
                time.sleep(0.8)
                continue

            obj = _extract_json(text)
            if obj and "classification" in obj:
                lab = obj["classification"].get("label", "")
                cf = float(obj["classification"].get("confidence", 0.0) or 0.0)
                rs = obj["classification"].get("reason", "")
                if lab in DISPLAY_LABELS:
                    return {"label": lab, "confidence": cf, "reason": rs}

            last_error = "cannot_parse_json"
        except Exception as e:
            last_error = str(e)
            if LOG_GEMINI:
                print(f"[Gemini] exception attempt {attempt+1}: {e}")
            time.sleep(1)
            continue

    if LOG_GEMINI and last_error:
        print(f"[Gemini] final fallback, reason={last_error}")
    return None


async def classify_with_gemini(crop: Image.Image):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(_classify_with_gemini_sync, crop)
    )


# =========================
# API meta
# =========================
@app.get("/")
def root():
    return {
        "status": "✅ running",
        "num_classes": len(model.names),
        "display_labels": DISPLAY_LABELS_VI,
        "gemini_enabled": bool(
            GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"
        ),
        "gemini_model": GEMINI_MODEL,
        "max_gemini_checks": MAX_GEMINI_CHECKS,
    }


@app.get("/labels/meta")
def labels_meta():
    return {
        k: {
            "label_vi": DISPLAY_LABELS_VI[k],
            "icon": LABEL_ICONS_VI.get(k, {}).get("icon", ""),
            "color": LABEL_ICONS_VI.get(k, {}).get("color", "#999"),
        }
        for k in DISPLAY_LABELS_VI
    }


@app.get("/gemini/ping")
def gemini_ping():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return {"ok": False, "reason": "No API key"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Return only JSON: {\"pong\":true}"}],
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }
    try:
        r = requests.post(
            GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY}, json=payload, timeout=15
        )
        j = r.json()
        text = safe_get_text_from_gemini_response(j)
        return {"ok": r.status_code == 200 and bool(text), "raw": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================
# API predict
# =========================
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    bg: Optional[Literal["white", "transparent"]] = DEFAULT_BG,
):
    if file.content_type not in ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận JPEG/PNG/WEBP.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rỗng.")

    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh.")
    img = _resize_max_side(img)

    try:
        boxes, names, plotted = await yolo_predict(img)
    except Exception as e:
        print("[YOLO error]", e)
        return {
            "items": [],
            "detections": [],
            "image": _pil_to_base64_jpeg(img),
            "detail": "Lỗi YOLO. Trả ảnh đã xử lý để kiểm tra.",
        }

    if boxes is None or len(boxes) == 0:
        return {
            "items": [],
            "detections": [],
            "image": _pil_to_base64_jpeg(img),
            "detail": "Không phát hiện vật thể.",
        }

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()

    dets: List[dict] = []
    tasks = []

    gemini_debug = {
        "enabled": bool(GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"),
        "model": GEMINI_MODEL,
        "calls": 0,
        "errors": [],
    }

    # Quyết định xem box nào gửi Gemini
    for i, (b, cf, c) in enumerate(zip(xyxy, confs, clss)):
        yolo_label = names.get(int(c), str(int(c)))
        crop = _crop(img, b, pad=12)

        # chỉ gọi Gemini nếu bật và còn trong MAX_GEMINI_CHECKS
        if i < MAX_GEMINI_CHECKS and gemini_debug["enabled"]:
            if LOG_GEMINI:
                print(f"[Gemini] classify idx={i}, yolo_label={yolo_label}")
            gemini_debug["calls"] += 1
            tasks.append(classify_with_gemini(crop))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        dets.append(
            {
                "box": b.tolist(),
                "yolo_label": yolo_label,
                "yolo_conf": float(cf),
            }
        )

    gem_results = await asyncio.gather(*tasks)

    # Xử lý plotted thành PIL
    if isinstance(plotted, np.ndarray):
        if plotted.ndim == 3 and plotted.shape[2] == 3:
            img_plt = Image.fromarray(plotted[:, :, ::-1])
        elif plotted.ndim == 3 and plotted.shape[2] == 4:
            b, g, r, a = (
                plotted[:, :, 0],
                plotted[:, :, 1],
                plotted[:, :, 2],
                plotted[:, :, 3],
            )
            img_rgba = np.stack([r, g, b, a], axis=2)
            img_plt = Image.fromarray(img_rgba)
        else:
            img_plt = Image.fromarray(plotted)
    elif isinstance(plotted, Image.Image):
        img_plt = plotted
    else:
        raise RuntimeError("Không thể chuyển đổi đối tượng 'plotted' sang ảnh PIL")

    # Quyết định final labels
    for d, gres in zip(dets, gem_results):
        yolo_label = d["yolo_label"]
        yolo_conf = d["yolo_conf"]

        if gres and isinstance(gres, dict) and gres.get("label") in DISPLAY_LABELS:
            gem_label = gres["label"]
            gem_conf = float(gres.get("confidence", 0.0) or 0.0)
            reason = gres.get("reason", "")

            # nếu trùng label thì lấy max conf
            if gem_label == yolo_label:
                final_label, final_conf, source = gem_label, max(
                    yolo_conf, gem_conf
                ), "yolo+gemini"
            else:
                # nếu khác, chỉ override nếu Gemini có confidence đủ cao
                if gem_conf >= yolo_conf:
                    final_label, final_conf, source = gem_label, gem_conf, "gemini"
                else:
                    final_label, final_conf, source = (
                        yolo_label,
                        yolo_conf,
                        "yolo",
                    )

            d.update(
                {
                    "final_label": final_label,
                    "final_conf": final_conf,
                    "source": source,
                    "reason": reason,
                    "gemini_label": gem_label,
                    "gemini_conf": gem_conf,
                    "gemini_used": True,
                }
            )
        else:
            # fallback YOLO an toàn
            safe_label = yolo_label if yolo_label in DISPLAY_LABELS else "plastic_bottle"
            d.update(
                {
                    "final_label": safe_label,
                    "final_conf": yolo_conf,
                    "source": "yolo",
                    "reason": "",
                    "gemini_used": False,
                }
            )

        d["label_vi"] = DISPLAY_LABELS_VI.get(d["final_label"], d["final_label"])
        d["icon"] = LABEL_ICONS_VI.get(d["final_label"], {}).get("icon", "")
        d["color"] = LABEL_ICONS_VI.get(d["final_label"], {}).get("color", "#999")

    # Đếm vật liệu để trả items cho dashboard
    counts = Counter([d["final_label"] for d in dets])
    items = [
        {"name": k, "label": DISPLAY_LABELS_VI.get(k, k), "quantity": v}
        for k, v in counts.items()
    ]
    # sắp xếp theo tên tiếng Việt
    items.sort(key=lambda x: x["label"])

    # vẽ bbox
    draw = ImageDraw.Draw(img_plt)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = d["color"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{d['icon']} {d['label_vi']} ({d['final_conf']:.2f}, {d['source']})"
        draw.text((x1, max(0, y1 - 16)), text, fill=color)

    return {
        "items": items,
        "detections": dets,
        "image": _pil_to_base64_jpeg(img_plt),
        "debug": gemini_debug,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"[❌ Exception] {request.url.path}: {exc}")
    return PlainTextResponse("Đã xảy ra lỗi nội bộ.", status_code=500)
