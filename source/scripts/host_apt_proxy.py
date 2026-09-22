"""文件说明：宿主机侧 apt 转发代理（改进 #27）。

场景：容器无外网出口（直连与宿主代理节点都断），但容器能经
host.docker.internal 访问宿主机端口，而宿主机网络正常。
本脚本在宿主机 0.0.0.0:3128 起一个最小 HTTP 转发代理（stdlib，无依赖），
容器内 apt 以 `Acquire::http::Proxy=http://host.docker.internal:3128` 走
宿主机网络取包。仅转发普通 GET/HEAD（TUNA 支持 http 源，无需 CONNECT）。

用法: python scripts/host_apt_proxy.py [port]
"""

from __future__ import annotations

import socket
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# 宿主机 shell 常设 http_proxy 指向（可能已死的）本地代理；转发代理必须
# 直连，否则 502 会原样传染（实测教训）。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# DNS 解析在部分会话里间歇性失败（getaddrinfo 11001）；TUNA 的 IPv6 路由
# 也不稳。强制 IPv4 + 结果缓存 + 重试，绕开抖动路径。
_IP_CACHE: dict[str, tuple[str, float]] = {}
_IP_TTL = 300.0
# 系统解析器间歇失败时的引导 IP（2026-09-18 实测可用的 TUNA Web 节点）。
_BOOTSTRAP_IPS = {"mirrors.tuna.tsinghua.edu.cn": "101.6.15.130"}


def _resolve_ipv4(host: str) -> str:
    cached = _IP_CACHE.get(host)
    now = time.time()
    if cached and now - cached[1] < _IP_TTL:
        return cached[0]
    last_error: Exception | None = None
    for _ in range(4):
        try:
            info = socket.getaddrinfo(host, 80, family=socket.AF_INET)
            ip = info[0][4][0]
            _IP_CACHE[host] = (ip, now)
            return ip
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    bootstrap = _BOOTSTRAP_IPS.get(host)
    if bootstrap:
        _IP_CACHE[host] = (bootstrap, now)
        return bootstrap
    raise last_error if last_error else OSError(f"resolve {host} failed")


class ForwardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, include_body: bool) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        raw_target = self.path
        # 经代理的请求是绝对 URI（GET http://host/path）；普通请求是
        # origin 形式（/path）+ Host 头。两种都要正确解析，否则拼出
        # "hosthttp://host/path" 这类垃圾主机名（此前 502 的真因）。
        if raw_target.startswith(("http://", "https://")):
            parts = urlsplit(raw_target)
        else:
            parts = urlsplit(f"http://{self.headers['Host']}{raw_target}")
        try:
            ip = _resolve_ipv4(parts.hostname)
        except OSError as error:
            message = f"proxy resolve failed: {error}".encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            if include_body:
                self.wfile.write(message)
            return
        port = parts.port or 80
        url = f"http://{ip}:{port}{parts.path}"
        if parts.query:
            url += f"?{parts.query}"
        request = urllib.request.Request(url, data=body, method=self.command)
        request.add_header("Host", self.headers["Host"])
        for header in ("Range", "If-Modified-Since", "User-Agent"):
            value = self.headers.get(header)
            if value:
                request.add_header(header, value)
        try:
            with _OPENER.open(request, timeout=120) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() in ("server", "date", "connection", "transfer-encoding"):
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if include_body:
                    self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if include_body:
                self.wfile.write(payload)
        except Exception as error:  # noqa: BLE001
            message = f"proxy error: {error}".encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            if include_body:
                self.wfile.write(message)

    def do_GET(self) -> None:  # noqa: N802
        self._forward(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._forward(include_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self._forward(include_body=True)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3128
    server = ThreadingHTTPServer(("0.0.0.0", port), ForwardHandler)
    print(f"host apt proxy listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
