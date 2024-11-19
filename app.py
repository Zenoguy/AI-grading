from flask import Flask, request, jsonify, render_template
import os
from grading_pipeline import grade_submission, extract_text_from_pdf

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/grade', methods=['POST'])
def grade_papers():
    # Validate file uploads
    if 'question_paper' not in request.files or 'answer_paper' not in request.files:
        return jsonify({"error": "Both question paper and answer paper files are required"}), 400

    question_paper = request.files['question_paper']
    answer_paper = request.files['answer_paper']

    # Check file extensions
    if not allowed_file(question_paper.filename) or not allowed_file(answer_paper.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Save files locally
    question_path = os.path.join(app.config['UPLOAD_FOLDER'], question_paper.filename)
    answer_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_paper.filename)
    question_paper.save(question_path)
    answer_paper.save(answer_path)

    # Extract text from PDFs
    question_text = extract_text_from_pdf(question_path)
    answer_text = extract_text_from_pdf(answer_path)

    # If no text is extracted from PDFs
    if not question_text or not answer_text:
        return jsonify({"error": "Failed to extract text from the PDFs. Please check the file contents."}), 400

    # Grade papers using the grading pipeline
    results = grade_submission(question_text, answer_text)

    # Return results as JSON response
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)

