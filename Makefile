.PHONY: run dev ngrok test lint call

run:
	uvicorn src.app:app --host 0.0.0.0 --port 8080
dev:
	python app.py
ngrok:
	ngrok http 8080
test:
	pytest -q
call:
	python run_call.py coffee
