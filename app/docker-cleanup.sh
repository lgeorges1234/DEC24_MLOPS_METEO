#!/bin/bash

# docker-cleanup.sh
# Script to clean up Docker resources and free disk space

echo "====== Starting Docker Cleanup ======"

echo "Stopping all running containers..."
docker stop $(docker ps -a -q) || true

echo "Removing all containers..."
docker rm $(docker ps -a -q) || true

echo "Removing all images..."
docker rmi $(docker images -q) --force || true

echo "Removing all volumes..."
docker volume prune -f || true

echo "Removing all networks..."
docker network prune -f || true

echo "Performing a complete system prune..."
docker system prune -a --volumes -f || true

echo "====== Docker Cleanup Complete ======"
echo "Current Docker disk usage:"
docker system df

# Make script executable with: chmod +x docker-cleanup.sh