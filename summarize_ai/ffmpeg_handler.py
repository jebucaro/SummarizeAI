from ffmpeg.asyncio import FFmpeg
from ffmpeg import FFmpegError, FFmpegAlreadyExecuted
import subprocess
from pathlib import Path
from exceptions import TranscodeError


def check_ffmpeg_installation() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        return True

    except subprocess.CalledProcessError:
        return False


async def extract_audio_with_ffmpeg(
    input_video_path: Path, output_audio_path: Path
) -> None:
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(input_video_path))
        .output(str(output_audio_path), {"q:a": 0, "map": "a", "f": "mp3"})
    )

    try:
        await ffmpeg.execute()
    except (FFmpegAlreadyExecuted, FFmpegError) as e:
        raise TranscodeError(e)
