import logging
import os
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


def load_company_info(data_folder):
    company_info = ""
    for filename in os.listdir(data_folder):
        if filename.endswith(".md"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                md_content = file.read()
                company_info += markdown.markdown(md_content)
    return company_info


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    data_folder = "data"
    company_info = load_company_info(data_folder)
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are an AI interviewer conducting a Aptitude skills interview. Ask one question at a time, keep it professional and conversational. Ensure responses are realistic, brief, and to the point, like a real candidate would answer in an actual interview. "
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

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, I will be taking your Aptitude Skills Interview today", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
