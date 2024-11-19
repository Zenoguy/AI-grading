import re
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import PyPDF2

# Load the reference CSV
CSV_FILE = './data/reference.csv'
reference_data = pd.read_csv(CSV_FILE)

# Load the BERT model and tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# 1. Extract text from PDFs
def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    """
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return str(e)

# 2. Parse questions and marks from the question paper
def parse_questions_and_marks(question_text):
    """
    Parse questions and their marks from the question paper text.
    Assumes questions follow the pattern: Q<number>: <text> [<marks> marks]
    """
    pattern = r"(Q\d+):\s*(.*?)\s*\[(\d+)\s*marks\]"
    matches = re.findall(pattern, question_text)

    questions = []
    for match in matches:
        question_id, question_text, marks = match
        questions.append({
            "id": question_id,
            "text": question_text.strip(),
            "marks": int(marks)
        })

    return questions

# 3. Parse student answers from the answer paper
def parse_student_answers(answer_text):
    """
    Parse student answers from the answer paper text.
    Assumes answers follow the pattern: Q<number>: <answer>
    """
    pattern = r"(Q\d+):\s*(.+)"
    matches = re.findall(pattern, answer_text)

    student_answers = {}
    for match in matches:
        question_id, answer = match
        student_answers[question_id] = answer.strip()

    return student_answers

# 4. Compute semantic similarity
def compute_similarity(answer_text, ideal_text):
    """
    Compute cosine similarity between student answer and ideal answer.
    """
    # Tokenize and embed both texts
    inputs_ideal = tokenizer(ideal_text, return_tensors="pt", padding=True, truncation=True)
    inputs_answer = tokenizer(answer_text, return_tensors="pt", padding=True, truncation=True)

    # Generate embeddings
    with torch.no_grad():
        embedding_ideal = model(**inputs_ideal).pooler_output
        embedding_answer = model(**inputs_answer).pooler_output

    # Compute cosine similarity
    similarity = cosine_similarity(embedding_ideal, embedding_answer)[0][0]
    return similarity

# 5. Grading function
def grade_submission(question_text, answer_text):
    """
    Grade the student's answers and calculate total marks scored.
    """
    questions = parse_questions_and_marks(question_text)
    student_answers = parse_student_answers(answer_text)

    total_marks_awarded = 0
    results = []
    for question in questions:
        q_id = question["id"]
        q_text = question["text"]
        marks = question["marks"]
        s_answer = student_answers.get(q_id, "")

        # Match question to CSV reference
        ref_question = reference_data[
            reference_data["question_text"].str.contains(q_text, case=False, na=False)
        ]

        if not ref_question.empty:
            ideal_answer = ref_question.iloc[0]["ideal_answer"]
            weighted_keywords = ref_question.iloc[0]["weighted_keywords"].split(',')

            # Compute similarity score
            similarity_score = compute_similarity(s_answer, ideal_answer)

            # Calculate keyword match score
            keywords_present = [kw for kw in weighted_keywords if kw in s_answer]
            keyword_coverage_score = len(keywords_present) / len(weighted_keywords) if weighted_keywords else 1.0

            # Combine similarity and keyword scores
            final_score = similarity_score * 0.7 + keyword_coverage_score * 0.3

            # Calculate final marks
            marks_awarded = final_score * marks
            total_marks_awarded += marks_awarded

            # Convert numerical values to Python floats for JSON serialization
            marks_awarded = float(marks_awarded)
            similarity_score = float(similarity_score)
            keyword_coverage_score = float(keyword_coverage_score)

            # Add detailed feedback
            results.append({
                "question": q_text,
                "student_answer": s_answer,
                "marks_awarded": marks_awarded,
                "similarity_score": similarity_score,
                "keyword_coverage_score": keyword_coverage_score,
                "missing_keywords": [kw for kw in weighted_keywords if kw not in s_answer]
            })
        else:
            results.append({
                "question": q_text,
                "student_answer": s_answer,
                "marks_awarded": 0,
                "remarks": "Question not found in reference database."
            })

    # Return results and total marks scored
    return {
        "results": results,
        "total_marks_scored": float(total_marks_awarded),
        "total_possible_marks": sum(q["marks"] for q in questions)
    }


# File paths for question and answer papers
question_file_path = './uploads/test3q.pdf'  # Path to the question paper
answer_file_path = './uploads/test3a.pdf'   # Path to the answer paper

# Extract text from the PDFs
question_text = extract_text_from_pdf(question_file_path)
answer_text = extract_text_from_pdf(answer_file_path)

# Grade the submission
results = grade_submission(question_text, answer_text)

# Print the results
print(results)

