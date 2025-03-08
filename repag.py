import logging
import asyncio
import os
import markdown
import tempfile
from datetime import datetime
from fpdf import FPDF
import json
from livekit.plugins.openai import LLM
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, silero, turn_detector

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("interview-agent")
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

def load_company_knowledge():
    """Load company information from the knowledge base file."""
    file_path = os.path.join("data", "company_info.md")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return markdown.markdown(content)
    else:
        logger.warning(f"Knowledge base file {file_path} not found")
        return "No company information available."

class InterviewAgent(VoicePipelineAgent):
    """Comprehensive interview agent that handles the entire interview process."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.company_knowledge = load_company_knowledge()
        self.candidate_name = "Candidate"
        self.interview_data = {
            "candidate_name": self.candidate_name,
            "interview_date": datetime.now().strftime("%Y-%m-%d"),
            "stages": {},
            "overall_score": 0,
            "overall_feedback": "",
        }
        
    async def run_interview(self):
        """Run the complete interview process from welcome to closing."""
        logger.info("Starting interview process")
        
        # Welcome stage
        await self.welcome_stage()
        
        # Technical assessment stage
        await self.technical_stage()
        
        # Soft skills stage
        await self.soft_skills_stage()
        
        # Culture fit stage
        await self.culture_fit_stage()
        
        # Closing stage and generate report
        await self.closing_stage()
        
        # Generate and save PDF report
        await self.generate_pdf_report()
        
        logger.info("Interview completed successfully")
        return True
    
    async def welcome_stage(self):
        """Handle the welcome and introduction stage."""
        logger.info("Starting Welcome stage")
        
        # Welcome message
        welcome_message = "Hello and welcome to your interview. I'm your AI interviewer and I'll be guiding you through several stages including technical assessment, soft skills evaluation, and culture fit. First, could you please tell me your name?"
        await self.say(welcome_message)
        
        # Get candidate name
        name_response = await self.ctx.wait_for_participant_speech()
        
        # Extract name from response
        name_prompt = f"""
        The candidate just responded to "Could you please tell me your name?" with: "{name_response}"
        Extract just their name (first and last if provided) from their response.
        Respond with ONLY the name, nothing else. If you can't determine the name, respond with "Candidate".
        """
        
        extracted_name = await self.llm.generate(name_prompt)
        self.candidate_name = extracted_name.strip() if extracted_name.strip() else "Candidate"
        self.interview_data["candidate_name"] = self.candidate_name
        logger.info(f"Identified candidate as: {self.candidate_name}")
        
        # Share company information
        company_info_message = f"Thank you, {self.candidate_name}. Before we begin, I'd like to share some information about our company. {self.company_knowledge} Do you have any initial questions about our company?"
        await self.say(company_info_message)
        
        # Handle any questions about the company
        company_questions = await self.ctx.wait_for_participant_speech()
        
        if len(company_questions.strip()) > 10:  # If there's a substantial response
            answer_prompt = f"""
            The candidate asked about the company: "{company_questions}"
            
            Based on this company information:
            {self.company_knowledge}
            
            Answer their question concisely and professionally. If you don't have enough information, politely let them know.
            """
            
            company_answer = await self.llm.generate(answer_prompt)
            await self.say(company_answer)
        else:
            await self.say(f"Great, {self.candidate_name}. Let's get started with the interview then.")
        
        # Brief pause before next stage
        await asyncio.sleep(1)
        
    async def technical_stage(self):
        """Handle the technical assessment stage."""
        logger.info("Starting Technical stage")
        
        # Transition to technical stage
        transition_message = f"We'll now move to the technical assessment part of the interview, {self.candidate_name}. I'd like to understand your technical background and capabilities."
        await self.say(transition_message)
        
        # Technical question
        tech_question = "Could you explain one technical project you're most proud of and what specific technologies you used to build it?"
        await self.say(tech_question)
        
        # Wait for response
        tech_response = await self.ctx.wait_for_participant_speech()
        logger.info(f"Received technical response: {tech_response}")
        
        # Analyze response and provide feedback
        tech_analysis_prompt = f"""
        You are a senior technical interviewer. The candidate, {self.candidate_name}, was asked:
        "{tech_question}"
        
        Their response was:
        "{tech_response}"
        
        Please provide:
        1. A brief (2-3 sentence) evaluation of their technical skills based on this response
        2. A score from 1-10 (where 10 is excellent)
        3. A follow-up comment or question that shows you were listening
        
        Format your response as JSON:
        {{
            "evaluation": "your evaluation here",
            "score": number,
            "follow_up": "your follow-up here"
        }}
        """
        
        tech_analysis_json = await self.llm.generate(tech_analysis_prompt)
        try:
            tech_analysis = json.loads(tech_analysis_json)
            tech_evaluation = tech_analysis.get("evaluation", "Thank you for sharing your experience.")
            tech_score = tech_analysis.get("score", 5)
            tech_follow_up = tech_analysis.get("follow_up", "That's interesting. Let's continue with the next section.")
            
            # Store the data for report
            self.interview_data["stages"]["technical"] = {
                "question": tech_question,
                "response": tech_response,
                "evaluation": tech_evaluation,
                "score": tech_score
            }
            
            # Respond to candidate
            await self.say(f"{tech_follow_up}")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tech analysis JSON: {tech_analysis_json}")
            await self.say("Thank you for sharing your technical experience. That gives me a good understanding of your background.")
        
        # Brief pause before next stage
        await asyncio.sleep(1)
    
    async def soft_skills_stage(self):
        """Handle the soft skills evaluation stage."""
        logger.info("Starting Soft Skills stage")
        
        # Transition to soft skills stage
        transition_message = f"Now, let's shift to discussing your soft skills, {self.candidate_name}. These are equally important for success in our team."
        await self.say(transition_message)
        
        # Soft skills question
        soft_question = "How do you handle conflicts or disagreements within your team? Could you provide a specific example?"
        await self.say(soft_question)
        
        # Wait for response
        soft_response = await self.ctx.wait_for_participant_speech()
        logger.info(f"Received soft skills response: {soft_response}")
        
        # Analyze response and provide feedback
        soft_analysis_prompt = f"""
        You are an interviewer focusing on soft skills. The candidate, {self.candidate_name}, was asked:
        "{soft_question}"
        
        Their response was:
        "{soft_response}"
        
        Please provide:
        1. A brief (2-3 sentence) evaluation of their conflict resolution and teamwork skills
        2. A score from 1-10 (where 10 is excellent)
        3. A follow-up comment that shows you were listening
        
        Format your response as JSON:
        {{
            "evaluation": "your evaluation here",
            "score": number,
            "follow_up": "your follow-up here"
        }}
        """
        
        soft_analysis_json = await self.llm.generate(soft_analysis_prompt)
        try:
            soft_analysis = json.loads(soft_analysis_json)
            soft_evaluation = soft_analysis.get("evaluation", "Thank you for sharing your approach.")
            soft_score = soft_analysis.get("score", 5)
            soft_follow_up = soft_analysis.get("follow_up", "That's valuable insight. Let's move on.")
            
            # Store the data for report
            self.interview_data["stages"]["soft_skills"] = {
                "question": soft_question,
                "response": soft_response,
                "evaluation": soft_evaluation,
                "score": soft_score
            }
            
            # Respond to candidate
            await self.say(f"{soft_follow_up}")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse soft skills analysis JSON: {soft_analysis_json}")
            await self.say("Thank you for sharing how you handle conflicts. That gives me good insight into your interpersonal skills.")
        
        # Brief pause before next stage
        await asyncio.sleep(1)
    
    async def culture_fit_stage(self):
        """Handle the culture fit assessment stage."""
        logger.info("Starting Culture Fit stage")
        
        # Transition to culture fit stage
        transition_message = f"For our final assessment area, {self.candidate_name}, I'd like to understand how you might fit within our company culture."
        await self.say(transition_message)
        
        # Culture fit question
        culture_question = "What kind of work environment brings out your best performance, and why do you think you would thrive in our company culture?"
        await self.say(culture_question)
        
        # Wait for response
        culture_response = await self.ctx.wait_for_participant_speech()
        logger.info(f"Received culture fit response: {culture_response}")
        
        # Analyze response and provide feedback
        culture_analysis_prompt = f"""
        You are an interviewer evaluating culture fit. The candidate, {self.candidate_name}, was asked:
        "{culture_question}"
        
        Their response was:
        "{culture_response}"
        
        Based on this company information:
        {self.company_knowledge}
        
        Please provide:
        1. A brief (2-3 sentence) evaluation of their potential cultural fit with the company
        2. A score from 1-10 (where 10 is excellent)
        3. A follow-up comment that shows engagement with their response
        
        Format your response as JSON:
        {{
            "evaluation": "your evaluation here",
            "score": number,
            "follow_up": "your follow-up here"
        }}
        """
        
        culture_analysis_json = await self.llm.generate(culture_analysis_prompt)
        try:
            culture_analysis = json.loads(culture_analysis_json)
            culture_evaluation = culture_analysis.get("evaluation", "Thank you for sharing your preferences.")
            culture_score = culture_analysis.get("score", 5)
            culture_follow_up = culture_analysis.get("follow_up", "That's helpful to know. Thank you.")
            
            # Store the data for report
            self.interview_data["stages"]["culture_fit"] = {
                "question": culture_question,
                "response": culture_response,
                "evaluation": culture_evaluation,
                "score": culture_score
            }
            
            # Respond to candidate
            await self.say(f"{culture_follow_up}")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse culture fit analysis JSON: {culture_analysis_json}")
            await self.say("Thank you for sharing your thoughts on our company culture. It helps us understand how you might fit into our team.")
        
        # Brief pause before next stage
        await asyncio.sleep(1)
    
    async def closing_stage(self):
        """Handle the closing stage and final evaluation."""
        logger.info("Starting Closing stage")
        
        # Generate overall feedback
        overall_prompt = f"""
        You are a senior hiring manager reviewing interview results for {self.candidate_name}.
        
        Based on these evaluations:
        
        Technical Assessment: {self.interview_data.get("stages", {}).get("technical", {}).get("evaluation", "No data")}
        Technical Score: {self.interview_data.get("stages", {}).get("technical", {}).get("score", 0)}
        
        Soft Skills Assessment: {self.interview_data.get("stages", {}).get("soft_skills", {}).get("evaluation", "No data")}
        Soft Skills Score: {self.interview_data.get("stages", {}).get("soft_skills", {}).get("score", 0)}
        
        Culture Fit Assessment: {self.interview_data.get("stages", {}).get("culture_fit", {}).get("evaluation", "No data")}
        Culture Fit Score: {self.interview_data.get("stages", {}).get("culture_fit", {}).get("score", 0)}
        
        Please provide:
        1. A concise overall evaluation (3-4 sentences) of the candidate
        2. An overall score from 1-10
        3. A recommendation (Strongly Recommend, Recommend, Consider, Do Not Recommend)
        
        Format your response as JSON:
        {{
            "overall_evaluation": "your evaluation here",
            "overall_score": number,
            "recommendation": "your recommendation here"
        }}
        """
        
        overall_analysis_json = await self.llm.generate(overall_prompt)
        try:
            overall_analysis = json.loads(overall_analysis_json)
            overall_evaluation = overall_analysis.get("overall_evaluation", "Thank you for participating in this interview.")
            overall_score = overall_analysis.get("overall_score", 5)
            recommendation = overall_analysis.get("recommendation", "Consider")
            
            # Store the overall data for report
            self.interview_data["overall_feedback"] = overall_evaluation
            self.interview_data["overall_score"] = overall_score
            self.interview_data["recommendation"] = recommendation
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse overall analysis JSON: {overall_analysis_json}")
            self.interview_data["overall_feedback"] = "Thank you for participating in this interview."
            self.interview_data["overall_score"] = 5
            self.interview_data["recommendation"] = "Consider"
        
        # Closing message
        closing_message = f"""
        Thank you, {self.candidate_name}, for completing all stages of our interview process. I've gathered valuable insights about your technical skills, soft skills, and potential culture fit.
        
        I'll be generating a detailed report from our conversation. Someone from our hiring team will reach out to you with next steps soon. Do you have any final questions before we wrap up?
        """
        
        await self.say(closing_message)
        
        final_questions = await self.ctx.wait_for_participant_speech()
        
        if len(final_questions.strip()) > 10:  
            final_answer_prompt = f"""
            The candidate asked a final question: "{final_questions}"
            
            Based on this company information:
            {self.company_knowledge}
            
            And the fact that they've just completed an interview with stages on technical skills, soft skills, and culture fit,
            provide a helpful, concise response to their question. If you don't have enough information, politely let them know.
            """
            
            final_answer = await self.llm.generate(final_answer_prompt)
            await self.say(final_answer)
        else:
            await self.say(f"Thank you again, {self.candidate_name}. It was a pleasure speaking with you today. Have a great day!")
    
    async def generate_pdf_report(self):
        """Generate and save a PDF report of the interview."""
        logger.info("Generating PDF report")
        
        try:
            # Create PDF object
            pdf = InterviewReportPDF(self.interview_data)
            
            # Generate the report
            report_filename = f"interview_report_{self.candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = os.path.join("reports", report_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Save the PDF
            pdf.generate_report()
            pdf.output(report_path)
            logger.info(f"PDF report saved to {report_path}")
            
            # Log the report generation
            logger.info(f"Interview report for {self.candidate_name} generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")

class InterviewReportPDF(FPDF):
    """Custom PDF generator for interview reports."""
    
    def __init__(self, interview_data):
        super().__init__()
        self.interview_data = interview_data
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(left=15, top=15, right=15)
        
    def header(self):
        """Generate report header."""
        # Add company logo if available
        self.image('ICEBREAKERS.png', 10, 8, 33)
        
        # Set font and colors
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(0, 51, 102)  # Dark blue
        
        # Company name
        self.cell(0, 10, 'Company Interview Report', 0, 1, 'C')
        
        # Date and reference number
        self.set_font('helvetica', '', 10)
        self.cell(0, 10, f"Date: {self.interview_data['interview_date']}", 0, 1, 'R')
        
        # Line break
        self.ln(5)
    
    def footer(self):
        """Generate report footer."""
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add a chapter title."""
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(240, 240, 240)  # Light gray
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)
    
    def content_text(self, text):
        """Add content text."""
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 5, text)
        self.ln(5)
    
    def score_indicator(self, score):
        """Add a visual score indicator."""
        self.set_font('helvetica', 'B', 10)
        self.cell(30, 10, f"Score: {score}/10", 0, 0)
        
        # Draw score bar
        self.set_draw_color(0, 0, 0)
        self.set_fill_color(220, 220, 220)  # Light gray for background
        self.rect(65, self.get_y() + 3, 100, 5, 'F')
        
        # Color depends on score
        if score >= 8:
            self.set_fill_color(0, 153, 0)  # Green
        elif score >= 5:
            self.set_fill_color(255, 153, 0)  # Orange
        else:
            self.set_fill_color(204, 0, 0)  # Red
            
        self.rect(65, self.get_y() + 3, score * 10, 5, 'F')
        self.ln(10)
    
    def generate_report(self):
        """Generate the complete report."""
        self.add_page()
        
        # Candidate information
        self.chapter_title('Candidate Information')
        self.content_text(f"Name: {self.interview_data['candidate_name']}")
        self.content_text(f"Interview Date: {self.interview_data['interview_date']}")
        self.ln(5)
        
        # Technical assessment
        if 'technical' in self.interview_data.get('stages', {}):
            tech_data = self.interview_data['stages']['technical']
            self.chapter_title('Technical Assessment')
            self.content_text(f"Question: {tech_data.get('question', 'N/A')}")
            self.content_text(f"Response: {tech_data.get('response', 'N/A')}")
            self.content_text(f"Evaluation: {tech_data.get('evaluation', 'N/A')}")
            self.score_indicator(tech_data.get('score', 0))
        
        # Soft skills assessment
        if 'soft_skills' in self.interview_data.get('stages', {}):
            soft_data = self.interview_data['stages']['soft_skills']
            self.chapter_title('Soft Skills Assessment')
            self.content_text(f"Question: {soft_data.get('question', 'N/A')}")
            self.content_text(f"Response: {soft_data.get('response', 'N/A')}")
            self.content_text(f"Evaluation: {soft_data.get('evaluation', 'N/A')}")
            self.score_indicator(soft_data.get('score', 0))
        
        # Culture fit assessment
        if 'culture_fit' in self.interview_data.get('stages', {}):
            culture_data = self.interview_data['stages']['culture_fit']
            self.chapter_title('Culture Fit Assessment')
            self.content_text(f"Question: {culture_data.get('question', 'N/A')}")
            self.content_text(f"Response: {culture_data.get('response', 'N/A')}")
            self.content_text(f"Evaluation: {culture_data.get('evaluation', 'N/A')}")
            self.score_indicator(culture_data.get('score', 0))
        
        # Overall evaluation
        self.add_page()
        self.chapter_title('Overall Evaluation')
        self.content_text(self.interview_data.get('overall_feedback', 'No overall feedback provided.'))
        self.score_indicator(self.interview_data.get('overall_score', 0))
        
        # Recommendation
        self.chapter_title('Recommendation')
        recommendation = self.interview_data.get('recommendation', 'No recommendation provided.')
        self.set_font('helvetica', 'B', 12)
        
        # Set color based on recommendation
        if recommendation == "Strongly Recommend":
            self.set_text_color(0, 102, 0)  # Dark green
        elif recommendation == "Recommend":
            self.set_text_color(0, 153, 0)  # Green
        elif recommendation == "Consider":
            self.set_text_color(255, 153, 0)  # Orange
        else:
            self.set_text_color(204, 0, 0)  # Red
            
        self.cell(0, 10, recommendation, 0, 1, 'L')
        self.set_text_color(0, 0, 0)  # Reset to black
        
        # Strengths and areas for improvement
        strengths_weaknesses_prompt = f"""
        Based on these interview evaluations:
        
        Technical: {self.interview_data.get('stages', {}).get('technical', {}).get('evaluation', 'No data')}
        Soft Skills: {self.interview_data.get('stages', {}).get('soft_skills', {}).get('evaluation', 'No data')}
        Culture Fit: {self.interview_data.get('stages', {}).get('culture_fit', {}).get('evaluation', 'No data')}
        
        Please provide:
        1. Three key strengths of the candidate
        2. Two areas for improvement
        
        Format your response as JSON:
        {{
            "strengths": ["strength 1", "strength 2", "strength 3"],
            "improvements": ["improvement 1", "improvement 2"]
        }}
        """
        
        # This should be run before the report generation, but for demonstration,
        # we'll include placeholder data
        strengths = ["Technical expertise", "Communication skills", "Problem-solving ability"]
        improvements = ["Could improve leadership skills", "May need more experience with team projects"]
        
        self.chapter_title('Key Strengths')
        for strength in strengths:
            self.content_text(f"• {strength}")
        
        self.chapter_title('Areas for Improvement')
        for improvement in improvements:
            self.content_text(f"• {improvement}")

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting interview process for participant {participant.identity}")

    # Initialize and run the interview agent
    agent = InterviewAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2"),
        llm=LLM.with_groq(model="llama3-8b-8192"),
        tts=deepgram.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=1.0,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text="""
            You are a professional and friendly AI interviewer. Your job is to conduct a comprehensive interview
            covering technical skills, soft skills, and culture fit. Be warm but professional, ask clear questions,
            and provide thoughtful feedback. Keep your responses concise and engaging.
            
            Throughout the interview, you should:
            - Be attentive and responsive to the candidate's answers
            - Show genuine interest in their experiences and skills
            - Provide constructive feedback when appropriate
            - Guide the conversation naturally through the different interview stages
            - End with a clear summary and next steps
            """
        )
    )
    
    # Start the agent
    agent.start(ctx.room, participant)
    
    # Run the complete interview
    await agent.run_interview()
    
    # Stop the agent
    agent.stop()
    
    logger.info("Interview completed successfully")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )