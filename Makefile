.PHONY: help prefetch up down logs ps migrate revision \
        dev-api dev-worker dev-ui test typecheck

help:
	@echo "Docker (recommended):"
	@echo "  make prefetch     one-time, ONLINE: download models into model-cache volume"
	@echo "  make up           build + start redis/api/worker/ui (offline)"
	@echo "  make down         stop and remove containers"
	@echo "  make logs         tail all logs"
	@echo ""
	@echo "Local dev (bare processes in the uv venv; needs a local redis):"
	@echo "  make dev-api      uvicorn service.api:app --reload"
	@echo "  make dev-worker   celery -A service.celery_app worker"
	@echo "  make dev-ui       streamlit run ui/app.py"
	@echo ""
	@echo "DB migrations (SQLAlchemy + Alembic):"
	@echo "  make migrate                 alembic upgrade head"
	@echo "  make revision m='message'    autogenerate a new migration"
	@echo ""
	@echo "  make test / make typecheck"

# --- Docker ---------------------------------------------------------------
prefetch:
	docker compose --profile prefetch run --rm prefetch

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

# --- Local dev (uv venv) --------------------------------------------------
dev-api:
	uv run uvicorn service.api:app --reload --port 8000

dev-worker:
	uv run celery -A service.celery_app worker --loglevel=info --concurrency=2

dev-ui:
	uv run streamlit run ui/app.py

# --- Migrations -----------------------------------------------------------
migrate:
	uv run alembic upgrade head

revision:
	uv run alembic revision --autogenerate -m "$(m)"

# --- Quality --------------------------------------------------------------
test:
	uv run pytest

typecheck:
	uv run mypy --strict rag_ingestion service ui/app.py tests
