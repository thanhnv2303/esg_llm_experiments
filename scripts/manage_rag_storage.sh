#!/usr/bin/env bash
set -euo pipefail

# Manage the local LightRAG storage services via Docker Compose.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

COMPOSE_FILE="${PROJECT_ROOT}/local/docker-compose.rag-storage.yml"
ENV_FILE="${PROJECT_ROOT}/local/configs/rag_anything_storage.env"
PROJECT_NAME=${COMPOSE_PROJECT_NAME:-ragstorage}

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Compose file not found: ${COMPOSE_FILE}" >&2
  exit 1
fi

COMMAND=${1:-up}
shift $(( $# > 0 ? 1 : 0 ))

docker_compose() {
  COMPOSE_PROJECT_NAME="${PROJECT_NAME}" docker compose -f "${COMPOSE_FILE}" "$@"
}

maybe_source_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi
}

case "${COMMAND}" in
  up)
    maybe_source_env
    docker_compose up "$@"
    ;;
  down)
    docker_compose down "$@"
    ;;
  restart)
    maybe_source_env
    docker_compose up -d "$@"
    docker_compose ps
    ;;
  logs)
    docker_compose logs "$@"
    ;;
  follow-logs)
    docker_compose logs -f "$@"
    ;;
  ps|status)
    docker_compose ps "$@"
    ;;
  cleanup)
    docker_compose down -v "$@"
    ;;
  *)
    cat >&2 <<'USAGE'
Usage: manage_rag_storage.sh <command> [args]

Commands:
  up            Start the storage stack in the background (default)
  down          Stop containers but keep volumes
  restart       Restart containers (alias for up)
  logs          Show container logs
  follow-logs   Tail container logs
  ps|status     Show container status
  cleanup       Stop containers and remove volumes

Extra arguments are forwarded to the underlying docker compose command.
USAGE
    exit 1
    ;;
esac
