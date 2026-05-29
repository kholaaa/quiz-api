from flask import Flask, request, jsonify
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pytesseract
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

def load_image_from_request(file):
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img

def decode_qr(img):
    decoded = decode(img)
    if not decoded:
        return None
    payload = decoded[0].data.decode("utf-8")
    result = {"raw": payload, "part1": {}, "part2": {}, "title": ""}
    try:
        parts = payload.split("|")
        result["title"] = parts[0].strip()
        for part in parts[1:]:
            part = part.strip()
            if part.startswith("Part-I:"):
                for item in part.replace("Part-I:", "").strip().split():
                    kv = item.split("=")
                    if len(kv) == 2:
                        result["part1"][kv[0]] = kv[1]
            elif part.startswith("Part-II:"):
                for item in part.replace("Part-II:", "").strip().split():
                    kv = item.split("=")
                    if len(kv) == 2:
                        result["part2"][kv[0]] = kv[1]
    except Exception as e:
        result["parse_error"] = str(e)
    return result

def extract_student_info(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = 2
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        enlarged = cv2.resize(gray, (width, height))
        denoised = cv2.GaussianBlur(enlarged, (3, 3), 0)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        full_text = pytesseract.image_to_string(thresh)
        name = ""
        reg_no = ""
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'name' in line_lower:
                if ':' in line:
                    name = line.split(':', 1)[1].strip()
                elif i + 1 < len(lines) and lines[i+1].strip():
                    name = lines[i+1].strip()
            if 'reg' in line_lower or 'roll' in line_lower:
                if ':' in line:
                    reg_no = line.split(':', 1)[1].strip()
                elif i + 1 < len(lines) and lines[i+1].strip():
                    reg_no = lines[i+1].strip()
            if 'registration' in line_lower and '#' in line:
                parts = line.split('#')
                if len(parts) > 1:
                    reg_no = parts[1].strip()
        return {
            "name": name if name else "Not detected",
            "reg_no": reg_no if reg_no else "Not detected"
        }
    except Exception as e:
        return {"name": "OCR error", "reg_no": str(e)}

def read_bubbles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # More sensitive threshold
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)
        area = cv2.contourArea(c)
        
        # More flexible conditions
        if 0.7 <= ratio <= 1.4 and 100 <= area <= 5000:
            bubbles.append((x, y, w, h))
    
    # Sort bubbles (row by row)
    bubbles = sorted(bubbles, key=lambda b: (b[1] // 25, b[0]))
    
    options = ["A", "B", "C", "D"]
    
    def get_answer(row):
        if len(row) < 4:
            return None
        best = None
        best_fill = 0
        for idx, (x, y, w, h) in enumerate(row[:4]):
            roi = thresh[y:y+h, x:x+w]
            fill = cv2.countNonZero(roi) / float(w * h)
            if fill > best_fill:
                best_fill = fill
                best = idx
        if best_fill < 0.25:   # Lowered threshold
            return None
        return options[best] if best is not None else None

    rows = [bubbles[i:i+4] for i in range(0, len(bubbles), 4)]
    
    part1 = {}
    part2 = {}
    
    for i, row in enumerate(rows[:8]):
        part1[f"Q{i+1}"] = get_answer(row)
    
    for i, row in enumerate(rows[8:16]):
        part2[f"Q{i+1}"] = get_answer(row)
    
    return {"part1": part1, "part2": part2}

    def get_answer(row):
        best = None
        best_fill = 0
        for idx, (x, y, w, h) in enumerate(row):
            roi = thresh[y:y+h, x:x+w]
            fill = cv2.countNonZero(roi) / float(w * h)
            if fill > best_fill:
                best_fill = fill
                best = idx
        if best_fill < 0.3:
            return None
        return options[best] if best is not None else None

    rows = [bubbles[i:i+4] for i in range(0, len(bubbles), 4)]
    part1 = {}
    part2 = {}
    for i, row in enumerate(rows[:8]):
        part1[f"Q{i+1}"] = get_answer(row)
    for i, row in enumerate(rows[8:16]):
        part2[f"Q{i+1}"] = get_answer(row)
    return {"part1": part1, "part2": part2}

def grade(student_answers, answer_key):
    correct = incorrect = unattempted = 0
    breakdown = {}
    for part in ["part1", "part2"]:
        s = student_answers.get(part, {})
        a = answer_key.get(part, {})
        for q, ans in a.items():
            student_ans = s.get(q)
            key = f"{part}_{q}"
            if not student_ans:
                unattempted += 1
                breakdown[key] = "unattempted"
            elif student_ans == ans:
                correct += 1
                breakdown[key] = "correct"
            else:
                incorrect += 1
                breakdown[key] = "incorrect"
    total = correct + incorrect + unattempted
    pct = round(correct / total * 100, 1) if total > 0 else 0
    g = "A" if pct >= 90 else "B" if pct >= 80 else "C" if pct >= 70 else "D" if pct >= 60 else "F"
    return {
        "correct": correct,
        "incorrect": incorrect,
        "unattempted": unattempted,
        "score": f"{correct}/{total}",
        "percentage": pct,
        "grade": g,
        "breakdown": breakdown
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"})

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400
    img = load_image_from_request(request.files["image"])
    answer_key = decode_qr(img)
    if not answer_key:
        return jsonify({"error": "QR code not found on image"}), 400
    student_info = extract_student_info(img)
    student_answers = read_bubbles(img)
    grade_result = grade(student_answers, answer_key)
    return jsonify({
        "student": student_info,
        "answer_key": answer_key,
        "student_answers": student_answers,
        "grade": grade_result
    })

@app.route("/batch", methods=["POST"])
def batch():
    if "images" not in request.files:
        return jsonify({"error": "No images sent"}), 400
    files = request.files.getlist("images")
    all_results = []
    for file in files:
        img = load_image_from_request(file)
        answer_key = decode_qr(img)
        if not answer_key:
            continue
        student_info = extract_student_info(img)
        student_answers = read_bubbles(img)
        grade_result = grade(student_answers, answer_key)
        all_results.append({
            "Name": student_info.get("name", ""),
            "Reg No": student_info.get("reg_no", ""),
            "Score": grade_result["score"],
            "Percentage": grade_result["percentage"],
            "Grade": grade_result["grade"],
            "Correct": grade_result["correct"],
            "Incorrect": grade_result["incorrect"],
            "Unattempted": grade_result["unattempted"],
        })
    if not all_results:
        return jsonify({"error": "No valid quizzes found"}), 400
    df = pd.DataFrame(all_results)
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"output/results_{timestamp}.xlsx"
    df.to_excel(path, index=False)
    return jsonify({
        "message": f"Processed {len(all_results)} quizzes",
        "results": all_results,
        "file": path
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)