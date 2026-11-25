.PHONY: help setup dev-backend dev-frontend test build up down logs clean

PYTHON ?= python
NPM ?= npm

help:
	@echo "Lyftr AI Chatbot - Available targets:"
	@echo "  make setup         Install backend and frontend dependencies"
	@echo "  make dev-backend   Run FastAPI backend on :8000"
	@echo "  make dev-frontend  Run Vite frontend dev server on :5173"
	@echo "  make test          Run backend pytest suite and frontend TypeScript/Vite build"
	@echo "  make build         Build Docker images"
	@echo "  make up            Start full stack via Docker Compose (frontend :3000, backend :8000)"
	@echo "  make down          Stop Docker Compose stack"
	@echo "  make logs          Tail Docker Compose logs"
	@echo "  make clean         Remove Docker volumes and local build artifacts"

setup:
	cd backend && $(PYTHON) -m pip install -r requirements.txt
	cd frontend && $(NPM) install

dev-backend:
	cd backend && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && $(NPM) run dev

test:
	cd backend && $(PYTHON) -m pytest -v
	cd frontend && $(NPM) run build

build:
	docker compose build

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf frontend/dist backend/.pytest_cache
