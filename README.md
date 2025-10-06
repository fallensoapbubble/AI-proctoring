# AI Proctoring System

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed

### Run the Application

```bash
# Clone and navigate to project
git clone <repository-url>
cd AI-Proctoring-System

# Build and start the application
docker-compose up --build -d proctoring-app


# Build without cache (if needed)
docker-compose build --no-cache proctoring-app
docker-compose up -d proctoring-app

# Access the application
# Open browser: http://localhost:5000
```

### Check Status

```bash
# View container status
docker-compose ps proctoring-app

# View logs
docker-compose logs proctoring-app

# View logs with tail
docker-compose logs --tail=100 proctoring-app

# Follow logs (real-time)
docker-compose logs -f proctoring-app

# Health check
curl http://localhost:5000/health
```

### Stop the Application

```bash
# Stop specific service
docker-compose stop proctoring-app

# Stop and remove containers
docker-compose down proctoring-app
```

That's it! The AI proctoring system is now running with all ML models and detectors ready.
