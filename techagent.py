import logging
import asyncio
import os
import signal
import sys
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
logger = logging.getLogger("voice-agent")
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Global variable to track current interview stage
CURRENT_INTERVIEW_TYPE = os.environ.get("INTERVIEW_TYPE", "technical")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class InterviewAgent(VoicePipelineAgent):
    def __init__(self, agent_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = agent_type
        self.next_requested = False
        self.questions_asked = 0  # Initialize question counter

    async def process_transcript(self, text):
        """Process the transcript to check for "NEXT" command"""
        
        if "NEXT" in text.upper():
            logger.info(f"'NEXT' command detected in transcript: {text}")
            self.next_requested = True
            return True
        return False

    async def conduct_interview(self):
        """Main interview loop"""
        # Introduce the interview type
        await self.say(f"Hello, I am your {self.agent_type} interviewer. I'll be assessing your {self.agent_type} skills.")
        
        while not self.next_requested:
            # Listen for candidate's response
            response = await self.listen()
            
            # Check if NEXT was said
            if await self.process_transcript(response):
                await self.say(f"Understood. Ending the {self.agent_type} interview and moving to the next assessment.")
                return "NEXT"
            
            # Process the response with the LLM
            llm_response = await self.respond(response)
            self.questions_asked += 1  # Increment question counter

            # Format the response
            has_asked_two_questions = self.questions_asked >= 2
            structured_response = {
                "has_asked_two_questions": has_asked_two_questions,
                "response_text": llm_response
            }
            
            # Use structured_response for TTS
            tts_input = structured_response["response_text"]  # Only send the response text to TTS
            await self.say(tts_input)  # Pass the TTS input to self.say()


            # Check if the interview is finished
            if "interview is finished" in llm_response.lower():
                await self.say("Thank you for completing this section of the interview.")
                return "FINISHED"
        
        return "NEXT"

async def run_technical_interview(ctx: JobContext):
    """Run the technical interview agent"""
    logger.info("Starting technical interview")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined technical interview: {participant.identity}")
    
    agent = InterviewAgent(
        "technical",
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2"),
        llm=LLM.with_groq(model="llama3-8b-8192"),
        tts=deepgram.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=2.5,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are conducting a technical interview. Your job is to assess the candidate's technical skills. "
                "DON'T reveal all your questions at once. Ask ONE question at a time and wait for the candidate's response. "
                "Start with a general question about their technical background. "
                "Then follow up with more specific questions based on their response. "
                "You can ask about programming languages, algorithms, system design, or previous projects. "
                "Always provide constructive feedback after each response. "
                "If the candidate says 'NEXT', immediately end the interview. "
                "After 1 question, end with 'This technical interview is finished. Thank you for your time.' "
                "Be professional but friendly. Don't be too rigid in your evaluation."
            ),
        ),
    )
    
    agent.start(ctx.room, participant)
    result = await agent.conduct_interview()
    
    if result == "NEXT":
        logger.info("Technical interview terminated by user with 'NEXT' command")
        os.environ["INTERVIEW_TYPE"] = "soft_skills"
        # Disconnect and restart the service
        await ctx.disconnect()
        os.execv(sys.executable, ['python'] + sys.argv)
    else:
        logger.info("Technical interview completed normally")
        await ctx.say("The technical interview has concluded. Thank you for your participation.")
        await ctx.disconnect()

async def run_soft_skills_interview(ctx: JobContext):
    """Run the soft skills interview agent"""
    logger.info("Starting soft skills interview")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined soft skills interview: {participant.identity}")
    
    agent = InterviewAgent(
        "soft skills",
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2"),
        llm=LLM.with_groq(model="llama3-8b-8192"),
        tts=deepgram.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=2.5,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are conducting a soft skills interview. Your job is to assess the candidate's interpersonal and communication skills. "
                "DON'T reveal all your questions at once. Ask ONE question at a time and wait for the candidate's response. "
                "Ask about teamwork, conflict resolution, leadership, and adaptability. "
                "Start with a question about how they work in teams. "
                "Then ask follow-up questions based on their responses. "
                "Always provide constructive feedback after each response. "
                "If the candidate says 'NEXT', immediately end the interview. "
                "After 3-4 questions, end with 'This soft skills interview is finished. Thank you for your time.' "
                "Be conversational and engaging. Focus on real-world scenarios."
            ),
        ),
    )
    
    agent.start(ctx.room, participant)
    result = await agent.conduct_interview()
    
    if result == "NEXT":
        logger.info("Soft skills interview terminated by user with 'NEXT' command")
        # Set environment variable to transition to the next interview type
        os.environ["INTERVIEW_TYPE"] = "aptitude"
        # Disconnect and restart the service
        await ctx.disconnect()
        os.execv(sys.executable, ['python'] + sys.argv)
    else:
        logger.info("Soft skills interview completed normally")
        await ctx.say("The soft skills interview has concluded. Thank you for your participation.")
        await ctx.disconnect()

async def run_aptitude_interview(ctx: JobContext):
    """Run the aptitude interview agent"""
    logger.info("Starting aptitude interview")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined aptitude interview: {participant.identity}")
    
    agent = InterviewAgent(
        "aptitude",
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2"),
        llm=LLM.with_groq(model="llama3-8b-8192"),
        tts=deepgram.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=2.5,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are conducting an aptitude interview. Your job is to assess the candidate's problem-solving and critical thinking skills. "
                "DON'T reveal all your questions at once. Ask ONE question at a time and wait for the candidate's response. "
                "Ask questions that test logical reasoning, analytical thinking, and creative problem-solving. "
                "Start with a question about how they approach novel problems. "
                "Then ask follow-up questions based on their responses. "
                "Always provide constructive feedback after each response. "
                "If the candidate says 'NEXT', immediately end the interview. "
                "After 3-4 questions, end with 'This aptitude interview is finished. Thank you for your time.' "
                "Be thoughtful and listen carefully to their reasoning process."
            ),
        ),
    )
    
    agent.start(ctx.room, participant)
    result = await agent.conduct_interview()
    
    if result == "NEXT":
        logger.info("Aptitude interview terminated by user with 'NEXT' command")
        # This was the last interview type, so we'll reset to the beginning
        os.environ["INTERVIEW_TYPE"] = "technical"
        # Disconnect and restart the service
        await ctx.disconnect()
        os.execv(sys.executable, ['python'] + sys.argv)
    else:
        logger.info("Aptitude interview completed normally")
        await ctx.say("All interviews have concluded. Thank you for your participation.")
        await ctx.disconnect()

async def entrypoint(ctx: JobContext):
    """Main entrypoint that routes to the appropriate interview type"""
    interview_type = os.environ.get("INTERVIEW_TYPE", "technical")
    
    logger.info(f"Starting {interview_type} interview process")
    
    if interview_type == "technical":
        await run_technical_interview(ctx)
    elif interview_type == "soft_skills":
        await run_soft_skills_interview(ctx)
    elif interview_type == "aptitude":
        await run_aptitude_interview(ctx)
    else:
        logger.error(f"Unknown interview type: {interview_type}")
        await ctx.say("Sorry, there was an error with the interview configuration.")
        await ctx.disconnect()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
