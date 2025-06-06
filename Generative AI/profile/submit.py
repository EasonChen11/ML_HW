import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# collect all m4a files in the current directory only include name start with output_
# m4aFiles = [f for f in os.listdir() if f.startswith("output_") and f.endswith(".m4a")]

# m4aFiles = sorted(m4aFiles)
# remove 000 001 002 003 004 005 006 007 008 009
# m4aFiles = [f for f in m4aFiles if not f.startswith("000") and not f.startswith("001") and not f.startswith("002") and not f.startswith("003") and not f.startswith("004") and not f.startswith("005") and not f.startswith("006") and not f.startswith("007") and not f.startswith("008") and not f.startswith("009")]

m4aFiles = [ 'output_002.m4a', 'output_003.m4a', 'output_004.m4a', 'output_005.m4a', 'output_006.m4a', 'output_007.m4a', 'output_008.m4a']

# print(m4aFiles)
# # exit()
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

txtFiles = sorted([f for f in os.listdir() if f.startswith("output_") and f.endswith(".txt")])

with open("total.txt", "w") as outfile:
    for fname in txtFiles:
        with open(fname, "r") as infile:
            outfile.write(infile.read() + "\n")

with open("total.txt", "r") as file:
    text = file.read()
    print(len(text))





