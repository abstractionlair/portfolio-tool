# Portfolio Optimizer - Deployment Guide

**Status**: PLANNING  
**Priority**: HIGH (after web interface complete)  
**Estimated Time**: 1-2 days for basic deployment

## Deployment Options Analysis

### Option 1: Simple Cloud Hosting (Recommended for Start)

**Platform**: DigitalOcean App Platform or Railway
- **Cost**: ~$5-20/month
- **Pros**: Simple, managed, good for prototypes
- **Cons**: Less control, potential cold starts

**Platform**: Render.com
- **Cost**: Free tier available, ~$7/month for paid
- **Pros**: Great free tier, easy deployment
- **Cons**: Free tier has spin-down delays

### Option 2: VPS Deployment (Recommended for Production)

**Platform**: DigitalOcean Droplet, AWS EC2, or Linode
- **Cost**: ~$6-40/month depending on size
- **Pros**: Full control, better performance, can run everything
- **Cons**: More setup required

### Option 3: Container-Based (Recommended for Scale)

**Platform**: AWS ECS, Google Cloud Run, or Azure Container Instances
- **Cost**: Pay per use, ~$10-50/month
- **Pros**: Scalable, modern, good CI/CD
- **Cons**: More complex setup

### Option 4: Serverless (Not Recommended)

**Why Not**: Our optimization engine has long-running computations and needs persistent connections for WebSockets

## Recommended Architecture (VPS-Based)

```
Internet
    ↓
Cloudflare (Free CDN/SSL)
    ↓
Nginx (Reverse Proxy)
    ├── FastAPI Backend (Port 8000)
    ├── React Frontend (Port 3000 → Static Files)
    └── PostgreSQL Database (Port 5432)
```

## Required Components

### 1. Domain and DNS
- **Domain**: ~$12/year (Namecheap, Google Domains)
- **DNS**: Cloudflare (free tier includes CDN and SSL)

### 2. Server Requirements
**Minimum VPS Specs**:
- 2 vCPUs
- 4GB RAM
- 80GB SSD
- Ubuntu 22.04 LTS

**Why These Specs**:
- Optimization calculations are CPU-intensive
- Data caching needs memory
- Historical data storage needs disk

### 3. Database
**PostgreSQL** for:
- User portfolios
- Optimization results
- Cached market data
- User authentication

**Redis** (optional) for:
- Session management
- Real-time data caching
- WebSocket connections

### 4. Required Services

**Core Services**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: portfolio_optimizer
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres/portfolio_optimizer
      REDIS_URL: redis://redis:6379
      SECRET_KEY: ${SECRET_KEY}
      FRED_API_KEY: ${FRED_API_KEY}
    depends_on:
      - postgres
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      REACT_APP_API_URL: https://api.yourdomain.com

volumes:
  postgres_data:
  redis_data:
```

### 5. Environment Configuration

**Required Environment Variables**:
```bash
# .env.production
# Database
DATABASE_URL=postgresql://user:pass@localhost/portfolio_optimizer

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# API Keys
FRED_API_KEY=your-fred-api-key

# Email (for reports)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Monitoring
SENTRY_DSN=your-sentry-dsn
```

### 6. File Structure for Deployment

```
portfolio-optimizer/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   └── scripts/
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── build/          # Production build
│   └── src/
├── nginx/
│   └── default.conf    # Nginx configuration
├── docker-compose.yml
├── .env.production
└── deploy.sh          # Deployment script
```

### 7. Nginx Configuration

**File**: `nginx/default.conf`
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # API Backend
    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Frontend
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
}
```

## Deployment Steps

### 1. Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Nginx
sudo apt install nginx -y

# Setup firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Create app directory
sudo mkdir -p /opt/portfolio-optimizer
sudo chown $USER:$USER /opt/portfolio-optimizer
```

### 2. Database Setup

```bash
# Create database
docker-compose up -d postgres
docker exec -it postgres psql -U postgres -c "CREATE DATABASE portfolio_optimizer;"

# Run migrations
docker-compose run backend python -m src.db.migrate
```

### 3. Deployment Script

**File**: `deploy.sh`
```bash
#!/bin/bash
set -e

echo "Deploying Portfolio Optimizer..."

# Pull latest code
git pull origin main

# Build frontend
cd frontend
npm install
npm run build
cd ..

# Build and restart backend
docker-compose build backend
docker-compose up -d

# Run migrations
docker-compose exec backend python -m src.db.migrate

# Restart nginx
sudo nginx -s reload

echo "Deployment complete!"
```

### 4. SSL Setup (Let's Encrypt)

```bash
# Install Certbot
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot

# Get certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

## Performance Optimization

### 1. Caching Strategy
- **CloudFlare**: CDN for static assets
- **Redis**: Cache market data (5-minute TTL)
- **PostgreSQL**: Cache optimization results
- **Frontend**: Service worker for offline capability

### 2. Background Jobs
Use Celery for:
- Portfolio rebalancing calculations
- Daily data updates
- Email reports
- Large optimizations

### 3. Monitoring

**Essential Monitoring**:
- **Uptime**: UptimeRobot (free)
- **Errors**: Sentry (free tier)
- **Performance**: New Relic or DataDog
- **Logs**: ELK stack or CloudWatch

**Health Checks**:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": check_database(),
        "redis": check_redis(),
        "data_provider": check_market_data()
    }
```

## Security Considerations

### 1. API Security
- Rate limiting (via Nginx or FastAPI)
- JWT authentication
- CORS properly configured
- Input validation on all endpoints

### 2. Database Security
- Encrypted connections
- Least privilege access
- Regular backups
- No credentials in code

### 3. Server Security
- Regular updates
- Fail2ban for SSH
- Firewall rules
- Docker security scanning

## Cost Estimation

### Monthly Costs (Production)
- **VPS**: $20-40 (DigitalOcean/Linode)
- **Domain**: ~$1 (annual cost divided)
- **Email**: $0 (using SMTP relay)
- **Monitoring**: $0-50 (depending on needs)
- **Backup Storage**: $5 (S3 or Backblaze)
- **Total**: ~$26-96/month

### Free Tier Option
- **Render.com**: Free backend (with limitations)
- **Vercel**: Free frontend hosting
- **Supabase**: Free PostgreSQL
- **Total**: $0/month (with limitations)

## Scaling Considerations

### When to Scale
- > 100 concurrent users
- > 10GB market data cache
- > 1000 optimizations/day

### How to Scale
1. **Horizontal**: Add more backend instances
2. **Database**: Read replicas for analytics
3. **Caching**: Dedicated Redis cluster
4. **CDN**: CloudFlare Pro for better caching

## Backup Strategy

### Daily Backups
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups"

# Database backup
docker-compose exec -T postgres pg_dump -U postgres portfolio_optimizer > $BACKUP_DIR/db_$DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/db_$DATE.sql s3://your-backup-bucket/

# Keep only last 30 days
find $BACKUP_DIR -name "db_*.sql" -mtime +30 -delete
```

## Deployment Checklist

### Pre-Deployment
- [ ] Domain purchased and configured
- [ ] Server provisioned
- [ ] SSL certificates obtained
- [ ] Environment variables set
- [ ] Database migrations tested
- [ ] API keys obtained (FRED, etc.)

### Deployment
- [ ] Code deployed
- [ ] Database migrated
- [ ] Static files served
- [ ] SSL working
- [ ] Health checks passing

### Post-Deployment
- [ ] Monitoring active
- [ ] Backups scheduled
- [ ] Error tracking enabled
- [ ] Performance baseline established
- [ ] Documentation updated

## Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
cp .env.example .env.production
# Edit .env.production with your values

# Deploy
docker-compose up -d

# View logs
docker-compose logs -f backend

# Update
git pull && docker-compose build && docker-compose up -d
```

This deployment guide provides a production-ready path from development to a live, scalable application!
