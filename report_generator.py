import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors
import os

def generate_radar_chart(scores, labels, filename):
    """Generates a radar chart (spider chart) and saves it as an image."""
    num_vars = len(labels)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the circle
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "polar"})
    ax.fill(angles, scores, color='blue', alpha=0.3)
    ax.plot(angles, scores, color='blue', linewidth=2)

    # Labels and Formatting
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="black")

    plt.title("Performance Overview", fontsize=11, fontweight="bold", color="#2C3E50")
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    plt.close()

def generate_pdf_report(
    filename, candidate_name, role, tech_result, aptitude_result, soft_skills_result, culture_fit_result, summary,
    tech_score, aptitude_score, soft_skills_score, culture_fit_score
):
    """Generates a structured PDF evaluation report with aligned content."""
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=0.5 * inch)
    styles = getSampleStyleSheet()

    # Define styles with Helvetica font
    title_style = ParagraphStyle('title_style', fontSize=16, spaceAfter=12, textColor=colors.black, alignment=1, fontName="Helvetica-Bold")
    section_title_style = ParagraphStyle('section_title', fontSize=12, spaceAfter=6, textColor=colors.black, fontName="Helvetica-Bold")
    content_style = ParagraphStyle('content_style', fontSize=10, spaceAfter=6, textColor=colors.black, fontName="Helvetica")

    elements = []

    # Report Title
    elements.append(Paragraph("Candidate Evaluation Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Candidate Info Table
    candidate_info = [
        [Paragraph("<b>Candidate Name:</b>", content_style), Paragraph(candidate_name, content_style)],
        [Paragraph("<b>Role Applied For:</b>", content_style), Paragraph(role, content_style)]
    ]
    elements.append(Table(candidate_info, colWidths=[2.5 * inch, 4.5 * inch]))
    elements.append(Spacer(1, 0.3 * inch))

    # Generate Radar Chart
    chart_filename = "performance_radar_chart.png"
    generate_radar_chart(
        [tech_score, aptitude_score, soft_skills_score, culture_fit_score], 
        ["Tech", "Aptitude", "Soft Skills", "Culture Fit"], 
        chart_filename
    )

    # Image and Summary Side by Side
    image = Image(chart_filename, width=2.0 * inch, height=2.0 * inch)
    summary_text = Paragraph(f"<b>Summary:</b> {summary}", content_style)
    
    # Table to align image & summary
    summary_table = Table(
        [[image, summary_text]], 
        colWidths=[2.0 * inch, 4.8 * inch], 
        rowHeights=[2.0 * inch]
    )

    # Style to keep summary aligned at the top
    summary_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 0.2 * inch))  # Minimal spacer

    # Compute Overall Average Score
    overall_avg_score = round((tech_score + aptitude_score + soft_skills_score + culture_fit_score) / 4, 2)

    # Scores Table
    scores_table = Table(
        [
            ["Category", "Score"],
            ["Technical Skills", f"{tech_score}/100"],
            ["Aptitude", f"{aptitude_score}/100"],
            ["Soft Skills", f"{soft_skills_score}/100"],
            ["Culture Fit", f"{culture_fit_score}/100"],
            ["Overall Average", f"{overall_avg_score}/100"]
        ],
        colWidths=[3.5 * inch, 2 * inch]
    )
    scores_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (-1, -1), (-1, -1), colors.lightblue)  # Highlight Overall Average
    ]))
    elements.append(scores_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Detailed Evaluation Sections
    sections = [
        ("Technical Round", tech_result),
        ("Aptitude Round", aptitude_result),
        ("Soft Skills Round", soft_skills_result),
        ("Culture Fit Round", culture_fit_result)
    ]
    
    for title, text in sections:
        elements.append(Paragraph(f"<b>{title}</b>", section_title_style))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(text, content_style))
        elements.append(Spacer(1, 0.3 * inch))

    # Build PDF
    doc.build(elements)
    
    # Clean up chart file
    os.remove(chart_filename)
    print(f"PDF evaluation report '{filename}' has been successfully generated.")

# Example usage
if __name__ == "__main__":
    generate_pdf_report(
        "candidate_report.pdf",
        "John Doe",
        "Software Engineer",
        "The candidate demonstrated strong programming skills in Python and Java. They effectively solved algorithmic problems and optimized solutions. However, there is room for improvement in system design discussions.",
        "Performed well in logical reasoning and problem-solving tasks. Demonstrated a structured approach to tackling numerical and analytical problems, though speed can be improved.",
        "Great communication skills, actively participated in discussions, and provided clear, concise answers. Strong leadership potential and ability to work collaboratively in a team.",
        "The candidate aligns well with the company culture and values. They showed enthusiasm, adaptability, and a positive attitude towards feedback. A great fit for team-based projects.",
        "Overall, the candidate is a strong fit for the role, with minor improvements needed in system design. Would recommend moving to the next round of discussions.",
        85, 78, 40, 88
    )