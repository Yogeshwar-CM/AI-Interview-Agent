from report_generator import generate_pdf_report

# Test Data
filename = "test_candidate_report.pdf"
candidate_name = "John Doe"
role = "Software Engineer"
tech_result = "Strong in algorithms and data structures, but needs improvement in system design."
aptitude_result = "Excellent logical reasoning and problem-solving skills."
soft_skills_result = "Good communication, but needs to be more assertive in group discussions."
culture_fit_result = "Aligns well with company values but could be more collaborative."
summary = "Overall, a well-rounded candidate with strong technical skills and good aptitude. Recommended for next round."
tech_score = 85
aptitude_score = 90
soft_skills_score = 75
culture_fit_score = 80

# Generate PDF Report
generate_pdf_report(
    filename, candidate_name, role, tech_result, aptitude_result, soft_skills_result, culture_fit_result, summary,
    tech_score, aptitude_score, soft_skills_score, culture_fit_score
)

print("Test PDF generated successfully!")