.PHONY: help build up down logs test eval clean

help:
	@echo "SLM Swarm Evaluation System - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make build          Build all Docker containers"
	@echo "  make up             Start all services"
	@echo "  make down           Stop all services"
	@echo ""
	@echo "Usage:"
	@echo "  make test           Test coordinator endpoints"
	@echo "  make eval           Run evaluation on synthetic dataset"
	@echo "  make eval-samsum    Run evaluation on SAMSum benchmark"
	@echo "  make logs           Follow coordinator logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Remove volumes and results"
	@echo "  make restart        Restart all services"
	@echo "  make health         Check service health"

build:
	docker compose build

up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@make health

down:
	docker compose down

logs:
	docker compose logs -f coordinator

test:
	@echo "Testing coordinator health..."
	curl -s http://localhost:8000/health | jq
	@echo ""
	@echo "Testing swarm summarization..."
	curl -s -X POST http://localhost:8000/summarize \
		-H "Content-Type: application/json" \
		-d '{"id":"test_001","message":"Quick test: The meeting is rescheduled to Friday at 2 PM in Conference Room B.","mode":"tldr"}' | jq

eval:
	docker compose run --rm evaluator

eval-samsum:
	docker compose run --rm -e DATASET_PATH=/app/dataset/samsum.jsonl evaluator

eval-no-big:
	docker compose run --rm -e ENABLE_BIG_MODEL=false evaluator

health:
	@echo "Checking service health..."
	@docker compose ps
	@echo ""
	@curl -s http://localhost:8000/health | jq || echo "Coordinator not ready yet"

restart:
	docker compose restart

clean:
	@echo "Warning: This will remove all Docker volumes and results!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		rm -rf models/ results/*/; \
		echo "Cleaned up volumes and results"; \
	fi

prepare-samsum:
	cd scripts && python prepare_samsum.py --output ../dataset/samsum.jsonl --split test

aggregate:
	cd scripts && python aggregate_results.py --results-dir ../results

