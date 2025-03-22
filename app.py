from flask import Flask, request, jsonify, render_template, redirect
from covid import load_model, predict_image
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    model = load_model()
    
    try:
        result = predict_image(model, file)
    
        if result:
            return jsonify({"prediction": result})
        else:
            return jsonify({"error": "Failed to analyze image"})
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Error during analysis: {str(e)}"})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)