#6d460ee96cb096dc201627657de69dfcb34373eb

import os
from dotenv import load_dotenv
import logging
from deepgram.utils import verboselogs
from datetime import datetime
import httpx

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()

AUDIO_FILE = "RES0206.mp3"

def transcribe_audio(file_path):
    try:
        # STEP 1 Create a Deepgram client using the API key in the environment variables
        config: DeepgramClientOptions = DeepgramClientOptions(
            verbose=False,
        )
        deepgram: DeepgramClient = DeepgramClient("6d460ee96cb096dc201627657de69dfcb34373eb", config)
        # OR use defaults
        # deepgram: DeepgramClient = DeepgramClient()

        # STEP 2 Call the transcribe_file method on the rest class
        with open(file_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        before = datetime.now()
        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        after = datetime.now()

        #print(response.to_json(indent=4))
        #print("")
        #print(response["results"]["channels"][0]["alternatives"][0]["transcript"])
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        #difference = after - before
        #print(f"time: {difference.seconds}")
        return transcript

    except Exception as e:
        print(f"Exception: {e}")

def main():
    try:
        # STEP 1 Create a Deepgram client using the API key in the environment variables
        config: DeepgramClientOptions = DeepgramClientOptions(
            verbose=False,
        )
        deepgram: DeepgramClient = DeepgramClient("6d460ee96cb096dc201627657de69dfcb34373eb", config)
        # OR use defaults
        # deepgram: DeepgramClient = DeepgramClient()

        # STEP 2 Call the transcribe_file method on the rest class
        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        before = datetime.now()
        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        after = datetime.now()

        #print(response.to_json(indent=4))
        #print("")
        print(response["results"]["channels"][0]["alternatives"][0]["transcript"])
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        difference = after - before
        print(f"time: {difference.seconds}")

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()