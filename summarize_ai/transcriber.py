import assemblyai as aai
from pathlib import Path
import asyncio
from exceptions import TranscribeError


async def transcribe_audio(input_audio_path: Path) -> aai.Transcript:
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        summarization=True,
        summary_model=aai.SummarizationModel.informative,
        summary_type=aai.SummarizationType.bullets_verbose,
    )

    future = transcriber.transcribe_async(str(input_audio_path), config)
    transcript = await asyncio.wrap_future(future)

    if transcript.error:
        raise TranscribeError(transcript.error)

    return transcript
