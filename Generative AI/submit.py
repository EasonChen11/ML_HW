import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("OCPP_GROQ_API_KEY"))


# collect all m4a files in the current directory only include name start with output_
m4aFiles = [f for f in os.listdir() if f.startswith("output_") and f.endswith(".m4a")]



m4aFiles = sorted(m4aFiles)
print(m4aFiles)

# exit()
# for filename in m4aFiles:
#     with open(filename, "rb") as file:
#         transcription = client.audio.transcriptions.create(
#             file=(filename, file.read()),
#             model="whisper-large-v3",
#             response_format="verbose_json",
#         )
#         print(transcription.text)
#         with open(filename.replace(".m4a", ".txt"), "w") as file:
#             file.write(transcription.text)

# txtFiles = sorted([f for f in os.listdir() if f.startswith("output_") and f.endswith(".txt")])

# with open("total.txt", "w") as outfile:
#     for fname in txtFiles:
#         with open(fname, "r") as infile:
#             outfile.write(infile.read() + "\n")

# with open("total.txt", "r") as file:
#     text = file.read()
#     print(len(text))





