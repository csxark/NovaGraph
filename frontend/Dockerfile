# =============================================================================
#  GraphRAG Research Assistant — Frontend Dockerfile (Two-Stage)
#
#  Stage 1 (build): Node 20 Alpine — install deps, compile Vite/React app
#  Stage 2 (serve): nginx Alpine   — serve the dist/ folder, proxy /api to backend
#
#  The final image is ~25 MB (no Node.js runtime in production layer).
# =============================================================================

# ── Stage 1: Node build ───────────────────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

# Copy manifest files first to leverage Docker layer caching
COPY package.json package-lock.json* ./

# Install all dependencies from lockfile (deterministic, no scripts from registry)
RUN npm ci --prefer-offline

# Copy the rest of the source code
COPY . .

# Compile the production bundle into /app/dist
RUN npm run build

# ── Stage 2: nginx serve ──────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS serve

LABEL org.opencontainers.image.title="GraphRAG Frontend"
LABEL org.opencontainers.image.description="React SPA served by nginx with /api proxy to backend"

# Remove the default nginx site config
RUN rm /etc/nginx/conf.d/default.conf

# Copy the compiled SPA from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy our custom nginx config (SPA routing + /api proxy)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# nginx exits with code 0 on reload; use daemon off so Docker tracks PID 1
CMD ["nginx", "-g", "daemon off;"]
