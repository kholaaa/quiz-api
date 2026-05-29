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
# QR DECODING  (multi-strategy, very robust)
# ─────────────────────────────────────────────

def _try_opencv_qr(img):
    """OpenCV's built-in QR detector — works well on photos."""
    detector = cv2.QRCodeDetector()
    data, pts, _ = detector.detectAndDecode(img)
    if data:
        return data
    # Try with WeChatQRCode if available (even better)
    try:
        wechat = cv2.wechat_qrcode_WeChatQRCode()
        texts, _ = wechat.detectAndDecode(img)
        if texts:
            return texts[0]
    except Exception:
        pass
    return None

def _try_pyzbar(img):
    """pyzbar — fast, but needs clean image."""
    results = pyzbar_decode(img)
    if results:
        return results[0].data.decode("utf-8")
    return None

def _preprocess_variants(img):
    """Generate multiple image variants to maximise decode chance."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = []

    # Original colour and gray
    variants.append(img)
    variants.append(gray)

    # Upscaled versions (2× and 3×)
    for scale in [2, 3]:
        variants.append(cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC))

    # CLAHE (fix uneven lighting / glare)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)
    variants.append(cv2.resize(enhanced, (w*2, h*2), interpolation=cv2.INTER_CUBIC))

    # Sharpened
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    variants.append(sharp)

    # Binary threshold (Otsu)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.resize(otsu, (w*2, h*2), interpolation=cv2.INTER_NEAREST))

    # Adaptive threshold
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    variants.append(adapt)

    # TOP-RIGHT CROP (QR is always top-right on this quiz sheet)
    for frac_h in [0.30, 0.40]:
        for frac_left in [0.50, 0.55, 0.60]:
            crop_colour = img[0:int(h*frac_h), int(w*frac_left):]
            crop_gray   = gray[0:int(h*frac_h), int(w*frac_left):]
            cw, ch = crop_colour.shape[1], crop_colour.shape[0]
            variants.append(crop_colour)
            variants.append(cv2.resize(crop_colour, (cw*3, ch*3), interpolation=cv2.INTER_CUBIC))
            variants.append(crop_gray)
            variants.append(cv2.resize(crop_gray,   (cw*3, ch*3), interpolation=cv2.INTER_CUBIC))

    return variants

def decode_qr(img):
    """Try every combination of preprocessor × decoder until one works."""
    variants = _preprocess_variants(img)

    for variant in variants:
        # Make sure it's uint8
        v = variant if variant.dtype == np.uint8 else variant.astype(np.uint8)

        # Try OpenCV QR first (most robust for photos)
        payload = _try_opencv_qr(v)
        if payload:
            return _parse_qr_payload(payload)

        # Then pyzbar
        payload = _try_pyzbar(v)
        if payload:
            return _parse_qr_payload(payload)

    return None

def _parse_qr_payload(payload):
    """
    Expected format:
      AI Quiz SP2026 Set-C | Part-I: Q1=D Q2=A Q3=B Q4=A Q5=D Q6=A Q7=A Q8=B | Part-II: Q1=C Q2=D ...
    """
    result = {"raw": payload, "part1": {}, "part2": {}, "title": ""}
    try:
        parts = payload.split("|")
        result["title"] = parts[0].strip()

        for part in parts[1:]:
            part = part.strip()
            prefix = ""
            if part.upper().startswith("PART-I:") or part.upper().startswith("PART I:"):
                prefix = "part1"
                body = re.sub(r"(?i)part-?i:", "", part).strip()
            elif part.upper().startswith("PART-II:") or part.upper().startswith("PART II:"):
                prefix = "part2"
                body = re.sub(r"(?i)part-?ii:", "", part).strip()
            else:
                continue

            for token in body.split():
                m = re.match(r"Q0*(\d+)=([A-Da-d])", token)
                if m:
                    q_num = m.group(1)           # "1".."8"  (leading zeros stripped)
                    answer = m.group(2).upper()
                    result[prefix][f"Q{q_num}"] = answer

    except Exception as e:
        result["parse_error"] = str(e)

    return result

# ─────────────────────────────────────────────
# STUDENT INFO OCR
# ─────────────────────────────────────────────

def extract_student_info(img):
    try:
        h, w = img.shape[:2]
        header = img[0:int(h * 0.32), :]
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3),
                               interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(enlarged, (3,3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        full_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')

        name = reg_no = ""
        lines = [l.strip() for l in full_text.split('\n') if l.strip()]
        for i, line in enumerate(lines):
            ll = line.lower()
            if ('name' in ll) and not name:
                if ':' in line:
                    candidate = line.split(':', 1)[1].strip()
                    if len(candidate) > 1:
                        name = candidate
                elif i+1 < len(lines):
                    name = lines[i+1]
            if ('reg' in ll or 'registration' in ll or 'roll' in ll) and not reg_no:
                if '#' in line:
                    reg_no = line.split('#', 1)[1].strip()
                elif ':' in line:
                    candidate = line.split(':', 1)[1].strip()
                    if len(candidate) > 1:
                        reg_no = candidate
                elif i+1 < len(lines):
                    reg_no = lines[i+1]

        return {
            "name":   name   if name   else "Not detected",
            "reg_no": reg_no if reg_no else "Not detected"
        }
    except Exception as e:
        return {"name": "OCR error", "reg_no": str(e)}

# ─────────────────────────────────────────────
# BUBBLE SHEET READING
# ─────────────────────────────────────────────

def read_bubbles(img):
    """
    Layout: Left half = Part-I (8 rows × 4 bubbles)
            Right half = Part-II (8 rows × 4 bubbles)
            Centre column = question numbers (text, ignored)
    """
    h, w = img.shape[:2]
    grid_top    = int(h * 0.28)
    grid_bottom = int(h * 0.96)

    left_end     = int(w * 0.42)
    right_start  = int(w * 0.58)

    part1_img = img[grid_top:grid_bottom, 0:left_end]
    part2_img = img[grid_top:grid_bottom, right_start:w]

    return {
        "part1": _read_half(part1_img),
        "part2": _read_half(part2_img),
    }

def _read_half(region):
    options = ["A", "B", "C", "D"]
    h, w = region.shape[:2]

    gray    = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh  = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    min_d = min(h, w) * 0.03
    max_d = min(h, w) * 0.20

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area  = cv2.contourArea(c)
        ratio = bw / float(bh) if bh > 0 else 0
        if 0.5 <= ratio <= 2.0 and min_d <= bw <= max_d and min_d <= bh <= max_d and area >= 60:
            bubbles.append((x, y, bw, bh))

    if len(bubbles) < 4:
        return _grid_fallback(thresh, h, w, options)

    row_h = h / 8.0
    rows  = {i: [] for i in range(8)}
    for b in bubbles:
        x, y, bw, bh = b
        cy = y + bh / 2
        rows[min(int(cy / row_h), 7)].append(b)

    result = {}
    for ri in range(8):
        qk  = f"Q{ri+1}"
        row = sorted(rows[ri], key=lambda b: b[0])
        if not row:
            result[qk] = None
            continue

        best_opt = best_fill = second_fill = 0
        best_idx = None
        for ci, (x, y, bw, bh) in enumerate(row[:4]):
            roi  = thresh[y:y+bh, x:x+bw]
            fill = cv2.countNonZero(roi) / float(bw*bh) if bw*bh > 0 else 0
            if fill > best_fill:
                second_fill = best_fill
                best_fill   = fill
                best_idx    = ci
            elif fill > second_fill:
                second_fill = fill

        if best_fill < 0.28:
            result[qk] = None
        elif best_fill - second_fill < 0.10 and second_fill > 0.20:
            result[qk] = "INVALID"
        else:
            result[qk] = options[best_idx] if best_idx is not None and best_idx < 4 else None

    return result

def _grid_fallback(thresh, h, w, options):
    result = {}
    row_h = h / 8
    col_w = w / 4
    for row in range(8):
        qk = f"Q{row+1}"
        best_f = 0
        best   = None
        for col in range(4):
            y1,y2 = int(row*row_h), int((row+1)*row_h)
            x1,x2 = int(col*col_w), int((col+1)*col_w)
            cell   = thresh[y1:y2, x1:x2]
            fill   = cv2.countNonZero(cell) / float(cell.size) if cell.size > 0 else 0
            if fill > best_f:
                best_f, best = fill, col
        result[qk] = options[best] if best is not None and best_f >= 0.28 else None
    return result

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
            if not correct_a:
                continue
            if not student_a or student_a == "INVALID":
                unattempted += 1
                breakdown[key] = "unattempted"
            elif student_a == correct_a:
                correct += 1
                breakdown[key] = "correct"
            else:
                incorrect += 1
                breakdown[key] = "incorrect"

    total  = correct + incorrect + unattempted
    marks  = correct - (0.25 * incorrect if negative_marking else 0)
    pct    = round(marks / total * 100, 1) if total > 0 else 0.0
    letter = ("A" if pct >= 90 else "B" if pct >= 80 else
              "C" if pct >= 70 else "D" if pct >= 60 else "F")

    return {
        "correct": correct, "incorrect": incorrect, "unattempted": unattempted,
        "score": f"{correct}/{total}", "percentage": pct,
        "grade": letter, "breakdown": breakdown,
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running ✓"})

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    img        = load_image(request.files["image"])
    answer_key = decode_qr(img)

    if not answer_key:
        student_info = extract_student_info(img)
        return jsonify({
            "error":   "QR code not found on image",
            "student": student_info,
            "tip":     "Ensure QR code is visible, well-lit and not blurry."
        }), 400

    student_info    = extract_student_info(img)
    student_answers = read_bubbles(img)
    neg_marking     = "negative" in answer_key.get("raw", "").lower()
    grade_result    = grade(student_answers, answer_key, neg_marking)

    return jsonify({
        "student":         student_info,
        "answer_key":      answer_key,
        "student_answers": student_answers,
        "grade":           grade_result,
    })

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
            errors.append({"file": file.filename, "error": "QR not found"})
            continue

        student_info    = extract_student_info(img)
        student_answers = read_bubbles(img)
        neg_marking     = "negative" in answer_key.get("raw", "").lower()
        grade_result    = grade(student_answers, answer_key, neg_marking)

        row = {
            "Quiz":        answer_key.get("title", ""),
            "Set":         answer_key.get("title", "").split("Set-")[-1] if "Set-" in answer_key.get("title","") else "",
            "Name":        student_info.get("name",""),
            "Reg No":      student_info.get("reg_no",""),
            "Correct":     grade_result["correct"],
            "Incorrect":   grade_result["incorrect"],
            "Unattempted": grade_result["unattempted"],
            "Total Marks": grade_result["correct"],
            "Percentage":  grade_result["percentage"],
            "Grade":       grade_result["grade"],
            "Score":       grade_result["score"],
        }
        for part, prefix in [("part1","Part1"),("part2","Part2")]:
            for q in range(1,9):
                row[f"{prefix}_Q{q:02d}"] = student_answers.get(part,{}).get(f"Q{q}","")
        all_results.append(row)

    if not all_results:
        return jsonify({"error": "No valid quizzes found", "details": errors}), 400

    df = pd.DataFrame(all_results)
    summary = {
        "Quiz": "SUMMARY",
        "Percentage":  round(df["Percentage"].mean(), 1),
        "Total Marks": f"Avg:{df['Total Marks'].mean():.1f} High:{df['Total Marks'].max()} Low:{df['Total Marks'].min()}",
        "Grade": "",
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"quiz_results_{timestamp}.xlsx"
    path      = f"output/{filename}"
    df.to_excel(path, index=False)

    return jsonify({
        "message": f"Processed {len(all_results)} quizzes",
        "results": all_results,
        "errors":  errors,
        "file":    filename,
    })

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    path = f"output/{filename}"
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)