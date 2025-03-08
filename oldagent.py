import logging
import asyncio
import os
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

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class InterviewAgent(VoicePipelineAgent):
    def __init__(self, agent_type, question, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = agent_type
        self.question = question

    async def conduct_interview(self):
        # Start the interview and ask the predefined question
        await self.say(f"Starting the {self.agent_type} interview.")
        await self.say(self.question)

        # Wait for the candidate's response
        response = await self.listen_and_respond()

        # Check if the LLM response contains "interview is finished"
        if "interview is finished" in response.lower():
            logger.info(f"{self.agent_type} interview marked as finished.")
            return  # Exit this agent and move to the next one

        # Provide feedback if "interview is finished" is not detected
        await self.say(f"Thank you for your response. Here's my feedback: {response}.")
        
        # Explicitly end the interview if not already done
        await self.say("Interview is finished. Thank you for your participation.")

class InterviewController:
    def __init__(self, ctx, participant):
        self.ctx = ctx
        self.participant = participant
        self.agents = []
        self.current_agent_index = 0

    def create_agents(self):
        common_args = {
            "vad": self.ctx.proc.userdata["vad"],
            "stt": deepgram.STT(model="nova-2"),
            "llm": LLM.with_groq(model="llama3-8b-8192"),
            "tts": deepgram.TTS(),
            "turn_detector": turn_detector.EOUModel(),
            "min_endpointing_delay": 0.5,
            "max_endpointing_delay": 2.5,
        }

        # Define three agents for Tech, Soft Skills, and Aptitude interviews with concise prompts
        self.agents = [
            InterviewAgent(
                "technical",
                "Explain one project you're proud of in a few sentences.",
                chat_ctx=llm.ChatContext().append(
                    role="system",
                    text=(
                        "You are conducting a technical interview. "
                        "Ask concise questions about programming skills or problem-solving. "
                        "When you have asked 1 question and gotten a response containing 'interview is finished', move to the next agent."
                    ),
                ),
                **common_args,
            ),
            InterviewAgent(
                "soft skills",
                "How do you resolve conflicts in a team? Keep it brief.",
                chat_ctx=llm.ChatContext().append(
                    role="system",
                    text=(
                        "You are assessing soft skills like communication and teamwork. "
                        "Ask clear questions and provide short feedback. "
                        "If 'interview is finished' is detected in a response, proceed to the next agent."
                    ),
                ),
                **common_args,
            ),
            InterviewAgent(
                "aptitude",
                "How would you solve a tough problem with limited resources?",
                chat_ctx=llm.ChatContext().append(
                    role="system",
                    text=(
                        "You are evaluating logical reasoning and problem-solving skills. "
                        "Ask focused questions and respond concisely. "
                        "If 'interview is finished' appears in a response, move to the next agent."
                    ),
                ),
                **common_args,
            ),
        ]

    async def run_interviews(self):
        for agent in self.agents:
            logger.info(f"Starting {agent.agent_type} interview.")
            agent.start(self.ctx.room, self.participant)

            # Conduct the interview for this agent
            await agent.conduct_interview()

            logger.info(f"{agent.agent_type} interview completed.")

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting interview process for participant {participant.identity}")

    # Initialize and run the InterviewController
    controller = InterviewController(ctx, participant)
    controller.create_agents()
    await controller.run_interviews()

    await ctx.say("Thank you for completing all interviews. We'll be in touch soon.")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
