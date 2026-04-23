FROM python:3.11-bookworm AS metadata-stage

WORKDIR /tmp

RUN --mount=type=bind,source=./.git/,target=/tmp/.git/ \
  git describe --tags --exact-match > /tmp/VERSION 2>/dev/null \
  || git rev-parse --short HEAD > /tmp/VERSION \
  && echo "Building version: $(cat /tmp/VERSION)"

FROM python:3.11-slim-bookworm

WORKDIR /app/zhenxun

ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1

EXPOSE 8080

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && \
    apt install -y --no-install-recommends curl fontconfig fonts-noto-color-emoji \
    && apt clean \
    && fc-cache -fv \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖声明文件，利用 Docker layer cache
COPY pyproject.toml uv.lock ./

# 安装依赖（--frozen 锁定版本，--no-install-project 不安装本项目，--no-dev 不安装开发依赖）
RUN uv sync --frozen --no-install-project --no-dev

# 复制应用代码
COPY . .

# 安装 Playwright 和 Chromium
RUN uv run playwright install --with-deps chromium \
  && rm -rf /var/lib/apt/lists/* /tmp/*

COPY --from=metadata-stage /tmp/VERSION /app/VERSION

VOLUME ["/app/zhenxun/data", "/app/zhenxun/resources", "/app/zhenxun/log"]

CMD ["uv", "run", "zx", "run"]
