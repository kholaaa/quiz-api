from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
import pytesseract
import pandas as pd
from datetime import datetime
import os
import re

app = Flask(__name__)

# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────

def load_image(file):
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img

# ─────────────────────────────────────────────
# QR DECODING
# ─────────────────────────────────────────────

def _try_opencv_qr(img):
    detector = cv2.QRCodeDetector()
    data, pts, _ = detector.detectAndDecode(img)
    if data:
        return data
    try:
        wechat = cv2.wechat_qrcode_WeChatQRCode()
        texts, _ = wechat.detectAndDecode(img)
        if texts:
            return texts[0]
    except Exception:
        pass
    return None

def _try_pyzbar(img):
    results = pyzbar_decode(img)
    if results:
        return results[0].data.decode("utf-8")
    return None

def _preprocess_variants(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = [img, gray]
    for scale in [2, 3]:
        variants.append(cv2.resize(img,  (w*scale, h*scale), interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)
    variants.append(cv2.resize(enhanced, (w*2, h*2), interpolation=cv2.INTER_CUBIC))
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    variants.append(cv2.filter2D(gray, -1, kernel))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.resize(otsu, (w*2, h*2), interpolation=cv2.INTER_NEAREST))
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(adapt)
    for frac_h in [0.30, 0.40]:
        for frac_left in [0.50, 0.55, 0.60]:
            crop_c = img [0:int(h*frac_h), int(w*frac_left):]
            crop_g = gray[0:int(h*frac_h), int(w*frac_left):]
            cw, ch = crop_c.shape[1], crop_c.shape[0]
            variants += [crop_c,
                         cv2.resize(crop_c, (cw*3, ch*3), interpolation=cv2.INTER_CUBIC),
                         crop_g,
                         cv2.resize(crop_g, (cw*3, ch*3), interpolation=cv2.INTER_CUBIC)]
    return variants

def decode_qr(img):
    for v in _preprocess_variants(img):
        v = v.astype(np.uint8)
        payload = _try_opencv_qr(v) or _try_pyzbar(v)
        if payload:
            return _parse_qr_payload(payload)
    return None

def _parse_qr_payload(payload):
    result = {"raw": payload, "part1": {}, "part2": {}, "title": ""}
    try:
        parts = payload.split("|")
        result["title"] = parts[0].strip()
        for part in parts[1:]:
            part = part.strip()
            if re.match(r"(?i)part-?i:", part) and not re.match(r"(?i)part-?ii:", part):
                prefix = "part1"
                body = re.sub(r"(?i)part-?i:", "", part).strip()
            elif re.match(r"(?i)part-?ii:", part):
                prefix = "part2"
                body = re.sub(r"(?i)part-?ii:", "", part).strip()
            else:
                continue
            for token in body.split():
                m = re.match(r"Q0*(\d+)=([A-Da-d])", token)
                if m:
                    result[prefix][f"Q{m.group(1)}"] = m.group(2).upper()
    except Exception as e:
        result["parse_error"] = str(e)
    return result

# ─────────────────────────────────────────────
# STUDENT INFO OCR
# ─────────────────────────────────────────────

def extract_student_info(img):
    try:
        h, w = img.shape[:2]
        header = img[0:int(h*0.32), :]
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3), interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(enlarged, (3,3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        full_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')
        name = reg_no = ""
        lines = [l.strip() for l in full_text.split('\n') if l.strip()]
        for i, line in enumerate(lines):
            ll = line.lower()
            if 'name' in ll and not name:
                if ':' in line:
                    c = line.split(':', 1)[1].strip()
                    if len(c) > 1: name = c
                elif i+1 < len(lines): name = lines[i+1]
            if ('reg' in ll or 'registration' in ll or 'roll' in ll) and not reg_no:
                if '#' in line:   reg_no = line.split('#',1)[1].strip()
                elif ':' in line:
                    c = line.split(':',1)[1].strip()
                    if len(c) > 1: reg_no = c
                elif i+1 < len(lines): reg_no = lines[i+1]
        return {"name": name or "Not detected", "reg_no": reg_no or "Not detected"}
    except Exception as e:
        return {"name": "OCR error", "reg_no": str(e)}

# ─────────────────────────────────────────────
# BUBBLE SHEET READING  — fixed for this layout
# ─────────────────────────────────────────────
#
# Layout (from the screenshot):
#   The answer grid has 9 columns:
#     col0=A  col1=B  col2=C  col3=D  | col4=QNo | col5=A  col6=B  col7=C  col8=D
#   8 rows (Q01–Q08)
#
# Strategy: find ALL bubble contours in the grid area,
# then assign each one to (row, col) by its (x,y) position.

def read_bubbles(img):
    h, w = img.shape[:2]

    # ── 1. Isolate the answer grid  ──────────────────────────────────────
    # Grid starts after header (~35% height) and goes to ~98%
    # Horizontally it spans the full width minus small margins
    grid_top    = int(h * 0.35)
    grid_bottom = int(h * 0.99)
    grid_left   = int(w * 0.02)
    grid_right  = int(w * 0.98)

    grid = img[grid_top:grid_bottom, grid_left:grid_right]
    gh, gw = grid.shape[:2]

    # ── 2. Threshold ─────────────────────────────────────────────────────
    gray    = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh  = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # ── 3. Find contours that look like bubbles ───────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area  = cv2.contourArea(c)
        ratio = bw / float(bh) if bh > 0 else 0
        # Bubbles are roughly square/circular, not too small, not too large
        min_size = min(gh, gw) * 0.03
        max_size = min(gh, gw) * 0.15
        if (0.5 <= ratio <= 2.0
                and min_size <= bw <= max_size
                and min_size <= bh <= max_size
                and area >= 100):
            cx = x + bw // 2
            cy = y + bh // 2
            # Compute fill ratio
            roi  = thresh[y:y+bh, x:x+bw]
            fill = cv2.countNonZero(roi) / float(bw * bh)
            bubbles.append((cx, cy, x, y, bw, bh, fill))

    if len(bubbles) < 16:
        # fallback to grid-based approach
        return _grid_fallback_full(thresh, gh, gw)

    # ── 4. Cluster into 8 rows by Y ───────────────────────────────────────
    # Sort by Y, cluster into 8 groups
    bubbles_sorted_y = sorted(bubbles, key=lambda b: b[1])

    # Find row boundaries using Y gaps
    rows_raw = []
    current_row = [bubbles_sorted_y[0]]
    for b in bubbles_sorted_y[1:]:
        if b[1] - current_row[-1][1] > gh * 0.06:
            rows_raw.append(current_row)
            current_row = [b]
        else:
            current_row.append(b)
    rows_raw.append(current_row)

    # Keep only rows with 4+ bubbles (skip header rows etc.)
    rows_raw = [r for r in rows_raw if len(r) >= 4]
    # Take at most 8 rows
    rows_raw = rows_raw[:8]

    # ── 5. For each row, split into Part-I (left 4) and Part-II (right 4) ─
    # The Q-number column divides left and right halves
    # Left half bubbles: x < 45% of grid width
    # Right half bubbles: x > 55% of grid width

    part1 = {}
    part2 = {}

    for ri, row in enumerate(rows_raw):
        qk = f"Q{ri+1}"

        # Sort row by X
        row_sorted = sorted(row, key=lambda b: b[0])

        left_bubbles  = [b for b in row_sorted if b[0] < gw * 0.45]
        right_bubbles = [b for b in row_sorted if b[0] > gw * 0.55]

        part1[qk] = _pick_answer(left_bubbles)
        part2[qk] = _pick_answer(right_bubbles)

    return {"part1": part1, "part2": part2}


def _pick_answer(bubbles_in_row):
    """Given up to 4 bubbles (sorted L→R = A,B,C,D), return the filled one."""
    options = ["A", "B", "C", "D"]
    if not bubbles_in_row:
        return None

    # Sort by X position (A=leftmost, D=rightmost)
    row = sorted(bubbles_in_row, key=lambda b: b[0])[:4]

    best_idx  = None
    best_fill = 0.0
    sec_fill  = 0.0

    for i, b in enumerate(row):
        fill = b[6]  # precomputed fill ratio
        if fill > best_fill:
            sec_fill  = best_fill
            best_fill = fill
            best_idx  = i
        elif fill > sec_fill:
            sec_fill = fill

    if best_fill < 0.28:
        return None  # unattempted
    if best_fill - sec_fill < 0.10 and sec_fill > 0.18:
        return "INVALID"  # two bubbles filled
    return options[best_idx] if best_idx is not None and best_idx < 4 else None


def _grid_fallback_full(thresh, gh, gw):
    """Divide entire grid into 8 rows × 8 cols (4 Part-I + 4 Part-II, skip centre)."""
    options = ["A","B","C","D"]
    part1 = {}
    part2 = {}
    row_h = gh / 8
    # Part-I: cols 0–3 (left 42% of width), Part-II: cols 5–8 (right 42%)
    p1_w = gw * 0.42
    p2_start = gw * 0.58
    p2_w = gw - p2_start
    col_w1 = p1_w / 4
    col_w2 = p2_w / 4

    for row in range(8):
        qk = f"Q{row+1}"
        y1, y2 = int(row*row_h), int((row+1)*row_h)

        best_f1, best1 = 0, None
        for col in range(4):
            x1 = int(col * col_w1)
            x2 = int((col+1) * col_w1)
            cell = thresh[y1:y2, x1:x2]
            f = cv2.countNonZero(cell) / float(cell.size) if cell.size > 0 else 0
            if f > best_f1: best_f1, best1 = f, col
        part1[qk] = options[best1] if best1 is not None and best_f1 >= 0.25 else None

        best_f2, best2 = 0, None
        for col in range(4):
            x1 = int(p2_start + col * col_w2)
            x2 = int(p2_start + (col+1) * col_w2)
            cell = thresh[y1:y2, x1:x2]
            f = cv2.countNonZero(cell) / float(cell.size) if cell.size > 0 else 0
            if f > best_f2: best_f2, best2 = f, col
        part2[qk] = options[best2] if best2 is not None and best_f2 >= 0.25 else None

    return {"part1": part1, "part2": part2}


# ─────────────────────────────────────────────
# GRADING
# ─────────────────────────────────────────────

def grade(student_answers, answer_key, negative_marking=False):
    correct = incorrect = unattempted = 0
    breakdown = {}
    for part in ["part1", "part2"]:
        s_part = student_answers.get(part, {})
        a_part = answer_key.get(part, {})
        for q_num in range(1, 9):
            qk        = f"Q{q_num}"
            key       = f"{part}_{qk}"
            student_a = s_part.get(qk)
            correct_a = a_part.get(qk)
            if not correct_a: continue
            if not student_a or student_a == "INVALID":
                unattempted += 1; breakdown[key] = "unattempted"
            elif student_a == correct_a:
                correct += 1;     breakdown[key] = "correct"
            else:
                incorrect += 1;   breakdown[key] = "incorrect"
    total  = correct + incorrect + unattempted
    marks  = correct - (0.25 * incorrect if negative_marking else 0)
    pct    = round(marks / total * 100, 1) if total > 0 else 0.0
    letter = "A" if pct>=90 else "B" if pct>=80 else "C" if pct>=70 else "D" if pct>=60 else "F"
    return {"correct":correct,"incorrect":incorrect,"unattempted":unattempted,
            "score":f"{correct}/{total}","percentage":pct,"grade":letter,"breakdown":breakdown}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"})

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400
    img        = load_image(request.files["image"])
    answer_key = decode_qr(img)
    if not answer_key:
        student_info = extract_student_info(img)
        return jsonify({"error":"QR code not found","student":student_info}), 400
    student_info    = extract_student_info(img)
    student_answers = read_bubbles(img)
    neg             = "negative" in answer_key.get("raw","").lower()
    grade_result    = grade(student_answers, answer_key, neg)
    return jsonify({"student":student_info,"answer_key":answer_key,
                    "student_answers":student_answers,"grade":grade_result})

@app.route("/batch", methods=["POST"])
def batch():
    if "images" not in request.files:
        return jsonify({"error": "No images sent"}), 400
    files       = request.files.getlist("images")
    all_results = []
    errors      = []
    for file in files:
        img        = load_image(file)
        answer_key = decode_qr(img)
        if not answer_key:
            errors.append({"file": file.filename, "error": "QR not found"}); continue
        student_info    = extract_student_info(img)
        student_answers = read_bubbles(img)
        neg             = "negative" in answer_key.get("raw","").lower()
        grade_result    = grade(student_answers, answer_key, neg)
        row = {
            "Quiz": answer_key.get("title",""),
            "Set":  answer_key.get("title","").split("Set-")[-1] if "Set-" in answer_key.get("title","") else "",
            "Name": student_info.get("name",""), "Reg No": student_info.get("reg_no",""),
            "Correct":grade_result["correct"],"Incorrect":grade_result["incorrect"],
            "Unattempted":grade_result["unattempted"],"Total Marks":grade_result["correct"],
            "Percentage":grade_result["percentage"],"Grade":grade_result["grade"],
            "Score":grade_result["score"],
        }
        for part, prefix in [("part1","Part1"),("part2","Part2")]:
            for q in range(1,9):
                row[f"{prefix}_Q{q:02d}"] = student_answers.get(part,{}).get(f"Q{q}","")
        all_results.append(row)
    if not all_results:
        return jsonify({"error":"No valid quizzes","details":errors}), 400
    df = pd.DataFrame(all_results)
    summary = {"Quiz":"SUMMARY","Percentage":round(df["Percentage"].mean(),1),
               "Total Marks":f"Avg:{df['Total Marks'].mean():.1f} High:{df['Total Marks'].max()} Low:{df['Total Marks'].min()}","Grade":""}
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    os.makedirs("output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"quiz_results_{ts}.xlsx"
    df.to_excel(f"output/{fn}", index=False)
    return jsonify({"message":f"Processed {len(all_results)} quizzes","results":all_results,"errors":errors,"file":fn})

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    path = f"output/{filename}"
    if not os.path.exists(path): return jsonify({"error":"File not found"}), 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)