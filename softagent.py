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
    return company_info


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    data_folder = "data"
    company_info = load_company_info(data_folder)
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are an AI interviewer conducting a soft skills interview. Ask one question at a time, keep it professional and conversational. Ensure responses are realistic, brief, and to the point, like a real candidate would answer in an actual interview. Dont speak too long responses please"
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

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, I will be taking your Soft Skills Interview today", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
