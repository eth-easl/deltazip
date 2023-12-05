import os
import json

def render_dialog(question, uncompressed_response, compressed_response):
    # replace linebreaks
    question = question.replace("\n", "[br]")
    uncompressed_response = uncompressed_response.replace("\n", "[br]")
    compressed_response = compressed_response.replace("\n", "[br]")
    # check if question contains non-ascii characters, such as
    if any([ord(c) >= 128 for c in question]) or any([ord(c) >= 128 for c in uncompressed_response]) or any([ord(c) >= 128 for c in compressed_response]):
        return None
    # check if contains latex special characters
    if any([c in ["&", "%", "$", "#", "_", "{", "}"] for c in question]) or any([c in ["&", "%", "$", "#", "_", "{", "}"] for c in uncompressed_response]) or any([c in ["&", "%", "$", "#", "_", "{", "}"] for c in compressed_response]):
        return None
    # check if contains "\"
    if any([c == "\\" for c in question]) or any([c == "\\" for c in uncompressed_response]) or any([c == "\\" for c in compressed_response]):
        return None
    # remove lyrics
    if "Lyrics" in question or "Lyrics" in uncompressed_response or "Lyrics" in compressed_response:
        return None
    template = """
    \dia[User]{<question>}
    \dia[Uncompressed]{<uncompressde_response>}
    \dia[Compressed]{<compressed_response>}
    \\vspace{0.1cm}
    \hrule\hrule
    \\vspace{0.1cm}
    """
    template = template.replace("<question>", question)
    template = template.replace("<uncompressde_response>", uncompressed_response)
    template = template.replace("<compressed_response>", compressed_response)
    return template


dialogs = []
with open("artifact/results/quality/chat/vicuna.jsonl", "r") as fp:
    data = [json.loads(line) for line in fp][1:]
    for datum in data:
        question = datum["question"]
        answers = datum["answers"]
        uncompressed_response = [answer for answer in answers if answer['from']=='original'][0]['text']
        compressed_response = [answer for answer in answers if answer['from']=='fmzip'][0]['text']
        dialogs.append(render_dialog(question, uncompressed_response, compressed_response))
dialogs = [dialog for dialog in dialogs if dialog is not None]
dialogs = dialogs[:100]
with open("artifact/results/quality/chat/vicuna_dialogs.tex", "w") as fp:
    fp.writelines(dialogs)    