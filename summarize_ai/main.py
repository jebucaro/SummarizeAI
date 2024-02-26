import streamlit as st
import subprocess
import assemblyai as aai
from pathlib import Path
from tempfile import NamedTemporaryFile
from exceptions import TranscodeError, TranscribeError
from ffmpeg import FFmpeg, FFmpegError, FFmpegAlreadyExecuted


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


def extract_audio_with_ffmpeg(input_video_path: Path, output_audio_path: Path) -> None:
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(input_video_path))
        .output(str(output_audio_path), {"q:a": 0, "map": "a", "f": "mp3"})
    )

    try:
        ffmpeg.execute()
    except (FFmpegAlreadyExecuted, FFmpegError) as e:
        raise TranscodeError(e)

    st.info(":musical_note: Audio extraction completed successfully.")


def transcribe_audio(input_audio_path: Path) -> list[aai.Paragraph]:
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(str(input_audio_path))

    if transcript.error:
        raise TranscribeError(transcript.error)

    st.info(":black_nib: Transcription completed successfully")

    return transcript.get_paragraphs()


def main():
    st.set_page_config(page_title="Summarize AI", page_icon=":robot_face:")
    st.markdown("# :robot_face: Summarize AI")

    aai.settings.api_key = st.secrets["assemblyai_api_key"]

    with st.form("my-form", clear_on_submit=True):
        dependency_installed = check_ffmpeg_installation()

        if dependency_installed:
            st.info(":gear: *ffmpeg* dependency is installed")
        else:
            st.warning(
                ":warning: *ffmpeg* not installed, please install it and refresh the page"
            )

        uploaded_files = st.file_uploader(
            label=":mag: Select one or multiple video files to transcribe.",
            type="mp4",
            accept_multiple_files=True,
            key="file_uploader_key",
            disabled=not dependency_installed,
        )

        submitted = st.form_submit_button(
            ":white_check_mark: Process", disabled=not dependency_installed
        )

        if submitted and uploaded_files:
            for uploaded_file in uploaded_files:
                with st.status(f"File: {uploaded_file.name}") as status:
                    temp_video_file = NamedTemporaryFile(delete=False)
                    temp_video_file_path = Path(temp_video_file.name)

                    try:
                        with open(temp_video_file_path, "wb") as file:
                            file.write(uploaded_file.getvalue())
                    except IOError as e:
                        st.error(e)
                    except Exception as e:
                        st.error(e)
                        temp_video_file.close()
                        continue

                    temp_audio_file = NamedTemporaryFile(suffix=".mp3", delete=False)
                    temp_audio_file_path = Path(temp_audio_file.name)

                    try:
                        extract_audio_with_ffmpeg(
                            temp_video_file_path, temp_audio_file_path
                        )
                    except TranscodeError as exception:
                        st.error(exception.message)
                        st.error(exception.arguments)
                        status.update(state="error", expanded=True)
                        continue
                    finally:
                        temp_video_file.close()

                    transcription = aai.Paragraph

                    try:
                        transcription = transcribe_audio(temp_audio_file_path)
                    except TranscribeError as e:
                        st.error(e)
                        status.update(state="error", expanded=True)
                        continue
                    finally:
                        temp_audio_file.close()

                    st.markdown("#### Transcription:")

                    with st.container():
                        for paragraph in transcription:
                            st.markdown(paragraph.text)

                    status.update(state="complete", expanded=False)

            st.info("All files were processed", icon=":partying_face:")


if __name__ == "__main__":
    main()
