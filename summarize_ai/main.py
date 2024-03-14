import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from io import BytesIO
from tempfile import NamedTemporaryFile
from pathlib import Path
import asyncio
import aiofiles.tempfile
from ffmpeg_handler import check_ffmpeg_installation, extract_audio_with_ffmpeg
from transcriber import transcribe_audio
from exceptions import TranscodeError, TranscribeError


semaphore = asyncio.Semaphore(3)


async def process_file(uploaded_file: BytesIO, status_placeholder: DeltaGenerator):
    async with semaphore:
        async with aiofiles.tempfile.NamedTemporaryFile(
            delete=False
        ) as temp_video_file:
            temp_video_file_path = Path(temp_video_file.name)
            await temp_video_file.write(uploaded_file.getvalue())

        temp_audio_file = NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio_file_path = Path(temp_audio_file.name)

        try:
            await extract_audio_with_ffmpeg(temp_video_file_path, temp_audio_file_path)
        except TranscodeError as e:
            status_placeholder.error(f":x: {e}")
            temp_video_file.close()
            temp_audio_file.close()
            return

        status_placeholder.info(
            ":musical_note: Audio extraction completed successfully."
        )

        try:
            transcript = await transcribe_audio(temp_audio_file_path)
        except TranscribeError as e:
            status_placeholder.error(f":x: {e}")
            return
        finally:
            temp_video_file.close()
            temp_audio_file.close()

        status_placeholder.info(":black_nib: Transcription completed successfully")

        status_placeholder.markdown("---")

        status_placeholder.markdown("### :page_facing_up: **Summary**")
        status_placeholder.markdown(transcript.summary)

        status_placeholder.markdown("---")

        status_placeholder.markdown("### :page_with_curl: **Transcription**")
        for paragraph in transcript.get_paragraphs():
            status_placeholder.markdown(paragraph.text)

        status_placeholder.update(state="complete", expanded=False)


async def main_async(uploaded_files: list[BytesIO]):
    status_placeholders = [
        st.status(f"File: {uploaded_file.name}") for uploaded_file in uploaded_files
    ]
    tasks = [
        process_file(uploaded_file, status_placeholder)
        for uploaded_file, status_placeholder in zip(
            uploaded_files, status_placeholders
        )
    ]
    await asyncio.gather(*tasks)


def main():
    st.set_page_config(page_title="Summarize AI", page_icon=":robot_face:")
    st.title(":robot_face: Summarize AI")

    dependency_installed = check_ffmpeg_installation()

    if dependency_installed:
        st.info(":gear: **ffmpeg** dependency is installed")
    else:
        st.warning(
            ":warning: **ffmpeg** not installed, please install it and refresh the page"
        )

    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            ":mag: Select one or multiple video files to transcribe.",
            type="mp4",
            accept_multiple_files=True,
            disabled=not dependency_installed,
        )

        submitted = st.form_submit_button(
            ":white_check_mark: Process", disabled=not dependency_installed
        )

        if submitted and uploaded_files:
            asyncio.run(main_async(uploaded_files))
        else:
            st.error("Please upload at least one file.")


if __name__ == "__main__":
    main()
