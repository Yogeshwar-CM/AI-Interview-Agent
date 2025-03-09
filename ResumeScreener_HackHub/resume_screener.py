import streamlit as st
import os
import re
import PyPDF2
import threading
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq

# Load environment variables
load_dotenv()

# Initialize AI model
model = Groq(id="llama-3.3-70b-versatile")

# Create AI Agent
agent = Agent(
    model=model,
    description="An AI agent that evaluates resumes based on job requirements, experience, and technical skills.",
)

# Email credentials from .env
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # Your Gmail
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # App Password

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_input):
    text = ''
    try:
        reader = PyPDF2.PdfReader(pdf_input)
        for page in reader.pages:
            text += page.extract_text() or ''
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Function to extract email from resume text
def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "Email not found"

# Function to extract candidate name from resume text
def extract_name(resume_text):
    lines = resume_text.strip().split("\n")
    return lines[0] if lines else "Candidate"  # Assume the first line contains the name

# Function to evaluate a resume against job requirements
def evaluate_resume(resume_text, job_requirements):
    prompt = f"""
    Check if the uploaded resume is a resume if not ask the user to upload a resume and stop. if not 
    Evaluate the candidate's resume against the following job requirements:

    {job_requirements}

    Provide a score for the following categories:
    1. Job Match: (1-10)
    2. Experience: (1-10)
    3. Technical Skills: (1-10)

    A candidate must meet or exceed these thresholds:
    - Job Match: 6
    - Experience: 3
    - Technical Skills: 7

    Provide a brief justification for each score and conclude with a line starting with 'Recommendation:' followed by 'Accept' or 'Reject'.
    If rejected, include constructive feedback for improvement.
    """
    
    response = agent.run(prompt + "\n\nResume:\n" + resume_text)
    return response.content

# Function to parse AI evaluation results
def parse_evaluation(evaluation_text):
    scores = {'Job Match': None, 'Experience': None, 'Technical Skills': None}
    recommendation = None
    feedback = None

    score_patterns = {
        'Job Match': r'Job Match:\s*([0-9]+)',
        'Experience': r'Experience:\s*([0-9]+)',
        'Technical Skills': r'Technical Skills:\s*([0-9]+)'
    }
    recommendation_pattern = r'Recommendation:\s*(Accept|Reject)'
    feedback_pattern = r'Recommendation: Reject\s*(.*)'

    for key, pattern in score_patterns.items():
        match = re.search(pattern, evaluation_text)
        if match:
            scores[key] = int(match.group(1))

    recommendation_match = re.search(recommendation_pattern, evaluation_text)
    if recommendation_match:
        recommendation = recommendation_match.group(1)

    feedback_match = re.search(feedback_pattern, evaluation_text, re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    return scores, recommendation, feedback

# Function to send email (Accepted/Rejected)
def send_email(to_email, candidate_name, decision, feedback=None):
    subject = f"Application Status - {decision}"

    if decision == "Accept":
        message_body = f"""
        Dear {candidate_name},

        Congratulations! After reviewing your resume, we are pleased to **{decision}** your application.
        Proceed to face verification by clicking the below link:
        https://faceverification-final.streamlit.app/

        Best Regards,  
        HR Team
        """
    else:
        message_body = f"""
        Dear {candidate_name},

        Thank you for applying. After reviewing your resume, we regret to inform you that we have **{decision}** your application.

        However, here’s some constructive feedback to help you improve your resume:
        {feedback if feedback else "Consider improving your technical skills and job alignment."}

        We encourage you to work on these areas and apply again in the future!

        Best Regards,  
        HR Team
        """

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message_body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        print(f"✅ Email sent to {to_email} ({decision})")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Streamlit UI for Resume Screening
st.title("📄 AI Resume Screener")
st.write("Upload a **Job Description** and **Candidate Resumes** for AI-driven evaluation.")

# Job description upload
st.subheader("📌 Upload Job Description (PDF)")
job_desc_uploaded = st.file_uploader("Upload Job Description", type=["pdf"])

if job_desc_uploaded:
    job_requirements_text = extract_text_from_pdf(job_desc_uploaded)
    st.write("### Extracted Job Description:")
    st.text(job_requirements_text)
else:
    st.warning("⚠ Please upload a job description PDF.")

# Resume uploads
st.subheader("📂 Upload Candidate Resumes (PDF)")
resume_uploaded = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

if job_desc_uploaded and resume_uploaded:
    results = []
    for resume_file in resume_uploaded:
        st.write(f"### Processing: {resume_file.name}")
        
        resume_text = extract_text_from_pdf(resume_file)
        email = extract_email(resume_text)
        candidate_name = extract_name(resume_text)

        # Evaluate resume
        evaluation_result = evaluate_resume(resume_text, job_requirements_text)
        st.write("#### Evaluation Result:")
        st.text(evaluation_result)

        # Extract scores, recommendation, and feedback
        scores, recommendation, feedback = parse_evaluation(evaluation_result)
        decision = recommendation if recommendation else "Rejected"

        # Send email notification
        if email != "Email not found":
            send_email(email, candidate_name, decision, feedback)

        results.append({
            "Candidate": candidate_name,
            "Email": email,
            "Job Match": scores["Job Match"],
            "Experience": scores["Experience"],
            "Technical Skills": scores["Technical Skills"],
            "Final Decision": decision
        })

    # Display results in a table
    if results:
        st.subheader("📊 Evaluation Summary")
        st.table(results)

# Function to run Face Verification
if __name__ == "__main__":
    threading.Thread(target=lambda: subprocess.run(["streamlit", "run", "face_verification.py", "--server.port=8502"]), daemon=True).start()
