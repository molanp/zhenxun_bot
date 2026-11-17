import json
from typing import Any, ClassVar, Literal

from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.decorators import Rules, tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolkitConfig, ToolResult
from zhenxun.services.ai.utils.logger import log_tool as logger
from zhenxun.services.network_proxy import ManagedAsyncClient


class RestApiToolkit(BaseToolkit):
    """
    通用 REST API 工具箱。
    让大模型无需编写任何代码即可调用外部 HTTP 接口获取或提交数据。
    """

    default_prefix = ""
    _default_config: ClassVar[ToolkitConfig] = ToolkitConfig(
        shared_options=Rules.superuser()
    )
    """默认仅超级用户可见，实例化时传入 shared_options 可覆盖"""

    default_instructions = (
        "你拥有调用外部 REST API 接口的能力。\n"
        "1. 使用 `make_request` 工具发起 HTTP 请求。\n"
        "2. 如果请求失败，请分析返回的错误信息。可能需要调整请求参数 (params) 或请求体 (json_data)。\n"  # noqa: E501
        "3. 务必根据返回的 JSON 数据，提取对用户有用的信息进行回答。"
    )

    def __init__(
        self,
        base_url: str = "",
        default_headers: dict[str, str] | None = None,
        default_params: dict[str, Any] | None = None,
        timeout: int = 30,
        **kwargs: Any,
    ):
        """
        初始化通用 REST API 工具箱。

        参数：
            base_url: 接口基础 URL，默认发起请求时将与其拼接。
            default_headers: 发起请求时携带的默认 Headers 请求头。
            default_params: 发起请求时携带的默认 URL 查询参数。
            timeout: 请求超时时间，单位为秒。
            kwargs: 其他透传给 BaseToolkit 的参数。
        """
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.default_headers = default_headers or {}
        self.default_params = default_params or {}
        self.timeout = timeout

    @tool(name="make_request")
    async def make_request(
        self,
        endpoint: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        context: RunContext | None = None,
    ) -> ToolResult:
        """
        发起 HTTP 请求获取外部数据。

        Args:
            endpoint: API的路径。初始化了base_url时只能请求其下的接口(相对路径或以base_url开头的完整URL)，否则需为以http(s)开头的完整URL。
            method: HTTP 请求方法。
            params: URL 查询参数字典。
            json_data: JSON 请求体字典。
        """  # noqa: E501
        url = endpoint
        if self.base_url and not endpoint.startswith(("http://", "https://")):
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if self.base_url and not (
            url == self.base_url or url.startswith(f"{self.base_url}/")
        ):
            return ToolResult(output=f"只能请求 {self.base_url} 下的接口").as_error()
        if not url.startswith(("http://", "https://")):
            return ToolResult(output="只能请求 http/https 地址").as_error()

        merged_params = self.default_params.copy()
        if params:
            merged_params.update(params)

        logger.info(f"🌐 [RestApiToolkit] 正在请求 {method} {url}")

        try:
            async with ManagedAsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=merged_params,
                    json=json_data,
                    headers=self.default_headers,
                )

            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = response.text

            result_dict = {"status_code": response.status_code, "data": response_data}

            is_error = response.status_code >= 400
            result = ToolResult(output=result_dict)

            if is_error:
                result = result.as_error(False, uncertain=response.status_code >= 500)
            return result

        except Exception as e:
            from zhenxun.services.ai.run.context import interrupt_transport

            interrupt_transport(e, "http_tool")
            logger.error(f"RestApiToolkit 结果未确认: {type(e).__name__}")
            return ToolResult(
                output="HTTP 操作结果未确认，请核查目标服务；本轮不自动重试。"
            ).as_error(False, uncertain=True)
