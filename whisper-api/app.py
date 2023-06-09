from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch
import os

if "WHISPER_MODEL" not in os.environ:
	whisper_model = "small"
else:
	whisper_model = os.environ["WHISPER_MODEL"]

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = whisper.load_model(whisper_model, download_root="/var/models", device=DEVICE)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handler():
	if not request.files:
		abort(400, "No file provided")
	
	results = []
	for filename, handle in request.files.items():
		temp = NamedTemporaryFile()
		handle.save(temp)
		result = model.transcribe(temp.name)
		results.append({
			"filename": filename,
			"language": result['language'],
			"transcript": result['text'],
		})
	return {"results": results}