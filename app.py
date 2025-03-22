import os
import base64
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import pytesseract
import easyocr
import torch
import re
import google.generativeai as genai
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Flaskアプリの作成
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# OCRモデルのセットアップ
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
reader = easyocr.Reader(["en"], gpu=False)

# Gemini APIのセットアップ
genai.configure(api_key="AIzaSyAB-pKWIWLuQcYc0mu2Z-cJHuWcs4X2UgE")
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


def save_base64_image(base64_str, filename):
    """Base64 画像データをデコードして保存"""
    base64_str = base64_str.split(",")[1]  # `data:image/png;base64,...` の `,` 以降を取得
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)
    return image_path


def preprocess_image(image_path):
    """画像の前処理"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    height, width = image.shape
    if min(height, width) < 400:
        image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)

    image = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
    image = cv2.medianBlur(image, 3)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return thresh


def correct_equation(equation: str) -> str:
    """数式の誤認識を補正"""
    corrections = {
        "O": "0", "l": "1", "—": "-", "−": "-", "Z": "2", "×": "*",
        "x+": "x + ", "x-": "x - ", "∞": "∞", "d x": "dx"
    }

    for wrong, correct in corrections.items():
        equation = equation.replace(wrong, correct)

    equation = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", equation)
    equation = equation.replace("x 2", "x^2").replace("X 2", "X^2")
    equation = equation.replace("^", "**")

    return equation.strip()


def extract_equation(image_path):
    """OCRで数式を抽出"""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None

    image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    pixel_values = ocr_processor(Image.fromarray(image_rgb), return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = ocr_model.generate(pixel_values)

    recognized_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return correct_equation(recognized_text)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        base64_image = request.form.get("cropped_image")
        if not base64_image:
            return render_template("index.html", error="画像がありません")

        filename = "cropped_image.png"
        image_path = save_base64_image(base64_image, filename)

        equation = extract_equation(image_path)
        if equation:
            return redirect(url_for("solve", equation=equation, image=image_path))
        else:
            return render_template("index.html", error="数式を認識できませんでした")

    return render_template("index.html")


@app.route("/solve", methods=["GET", "POST"])
def solve():
    equation = request.args.get("equation", "")
    image_path = request.args.get("image", "")

    if request.method == "POST":
        problem_statement = request.form["problem"]
        gemini_prompt = f"問題: {problem_statement}\n\n数式: {equation}\n\nこの問題を解いてください。"
        response = gemini_model.generate_content(gemini_prompt)

        return render_template("result.html", equation=equation, problem=problem_statement, solution=response.text)

    return render_template("solve.html", equation=equation, image=image_path)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
