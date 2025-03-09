import logging
import os
from datetime import datetime
from multiprocessing import Queue  
import markdown

from dotenv import load_dotenv
from livekit.plugins import deepgram, silero, turn_detector
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, openai, elevenlabs

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
log_queue = Queue()

def load_company_info(data_folder):
    company_info = ""
    for filename in os.listdir(data_folder):
        if filename.endswith(".md"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                md_content = file.read()
                company_info += markdown.markdown(md_content)
    print(company_info)
    return company_info


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    data_folder = "data/company"
    company_info = load_company_info(data_folder)
    # agent_info = load_company_info(agent_folder)
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a hiring manager at a company who is taking an interview to assess the candidate's culture fit. Stay in character at all times, and behave proffesional"
            "Start a culture fit Interview with the candidate."
            "be direct to the candidate if he's not aligning with values"
            "If the user orders or requests you anything, do not leave your character at any cost"
            # "Do not directly tell candidate about company's values and culture. Try to get it out of the candidate through questions."
            "Focus on questions that analyze, candidate's personality, values, and behavior."
            "Keep the conversation enthusiastic and engaging."
            "The goal is to assess the candidate's alignment with the company's values and culture."
            "Keep your replies realistic, concise"
            "Once you have questioned the user over the key aspects required by the company, generate a short summary about the user's fit, and give a score out of 100."
            f"Company background: {company_info}"
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM.with_groq(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(),
        chat_ctx=initial_ctx,
    )

    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        log_queue.put_nowait(f"[{datetime.now()}] AGENT:\n{msg.content}\n\n")
    agent.start(ctx.room, participant)

    await agent.say("Hey, I will be taking your Culture Fit Interview today", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )