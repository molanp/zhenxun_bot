"""
仓库文件管理器，用于从GitHub和阿里云CodeUp获取指定文件内容
"""

import contextlib
from pathlib import Path
import shutil
import tempfile
from typing import overload

from httpx import Response

from zhenxun.services.log import logger
from zhenxun.utils.github_utils import GithubUtils
from zhenxun.utils.github_utils.models import AliyunTreeType, GitHubStrategy, TreeType
from zhenxun.utils.http_utils import AsyncHttpx

from .config import LOG_COMMAND, RepoConfig
from .exceptions import (
    FileNotFoundError,
    GitUnavailableError,
    NetworkError,
    RepoManagerError,
)
from .models import FileDownloadResult, RepoFileInfo, RepoType
from .utils import get_aliyun_group_for_repo, prepare_aliyun_url, sparse_checkout_clone


class RepoFileManager:
    """仓库文件管理器，用于获取GitHub和阿里云仓库中的文件内容"""

    def __init__(self, config: RepoConfig | None = None):
        """
        初始化仓库文件管理器

        参数:
            config: 配置，如果为None则使用默认配置
        """
        self.config = config or RepoConfig.get_instance()
        self.config.ensure_dirs()

    @overload
    async def get_github_text_content(
        self, url: str, file_path: str, branch: str = "main", ignore_error: bool = False
    ) -> str: ...

    @overload
    async def get_github_text_content(
        self,
        url: str,
        file_path: list[str],
        branch: str = "main",
        ignore_error: bool = False,
    ) -> list[tuple[str, str]]: ...

    async def get_github_text_content(
        self,
        url: str,
        file_path: str | list[str],
        branch: str = "main",
        ignore_error: bool = False,
    ) -> str | list[tuple[str, str]]:
        """
        获取GitHub仓库文本文件内容

        参数:
            url: 仓库URL
            file_path: 文件路径或文件路径列表
            ignore_error: 是否忽略错误

        返回:
            list[tuple[str, str]]: 文件路径，文件内容
        """
        results: list[tuple[str, str]] = []
        is_str_input = isinstance(file_path, str)
        try:
            if is_str_input:
                file_path = [file_path]
            repo_info = GithubUtils.parse_github_url(url)
            repo_info.branch = branch
            if await repo_info.update_repo_commit():
                logger.info(f"获取最新提交: {repo_info.branch}", LOG_COMMAND)
            else:
                logger.warning(f"获取最新提交失败: {repo_info}", LOG_COMMAND)
            for f in file_path:
                try:
                    file_url = await repo_info.get_raw_download_urls(f)
                    for fu in file_url:
                        response: Response = await AsyncHttpx.get(
                            fu, check_status_code=200
                        )
                        if response.status_code == 200:
                            logger.info(f"获取github文件内容成功: {f}", LOG_COMMAND)
                            text_content = response.content.decode(
                                "utf-8", errors="ignore"
                            )
                            results.append((f, text_content))
                            break
                        else:
                            logger.warning(
                                f"获取github文件内容失败: {response.status_code}",
                                LOG_COMMAND,
                            )
                except Exception as e:
                    logger.warning(f"获取github文件内容失败: {f}", LOG_COMMAND, e=e)
                    if not ignore_error:
                        raise
        except Exception as e:
            logger.error(f"获取GitHub文件内容失败: {file_path}", LOG_COMMAND, e=e)
            raise
        logger.debug(f"获取GitHub文件内容: {[r[0] for r in results]}", LOG_COMMAND)

        return results[0][1] if is_str_input and results else results

    @overload
    async def get_aliyun_text_content(
        self,
        repo_name: str,
        file_path: str,
        branch: str = "main",
        ignore_error: bool = False,
    ) -> str: ...

    @overload
    async def get_aliyun_text_content(
        self,
        repo_name: str,
        file_path: list[str],
        branch: str = "main",
        ignore_error: bool = False,
    ) -> list[tuple[str, str]]: ...

    async def get_aliyun_text_content(
        self,
        repo_name: str,
        file_path: str | list[str],
        branch: str = "main",
        ignore_error: bool = False,
    ) -> str | list[tuple[str, str]]:
        """
        获取阿里云CodeUp仓库文本文件内容

        参数:
            repo: 仓库名称
            file_path: 文件路径
            branch: 分支名称
            ignore_error: 是否忽略错误
        返回:
            list[tuple[str, str]]: 文件路径，文件内容
        """
        results = []
        is_str_input = isinstance(file_path, str)
        # 导入阿里云相关模块
        from zhenxun.utils.github_utils.models import AliyunFileInfo

        if is_str_input:
            file_path = [file_path]
        for f in file_path:
            try:
                content = await AliyunFileInfo.get_file_content(
                    file_path=f, repo=repo_name, ref=branch
                )
                results.append((f, content))
            except Exception as e:
                if "code: 404" not in str(e):
                    logger.warning(
                        f"获取阿里云文件内容失败: {file_path}", LOG_COMMAND, e=e
                    )
                if not ignore_error:
                    raise
        logger.debug(f"获取阿里云文件内容: {[r[0] for r in results]}", LOG_COMMAND)
        return results[0][1] if is_str_input and results else results

    @overload
    async def get_text_content(
        self,
        repo_url: str,
        file_path: str,
        branch: str = "main",
        repo_type: RepoType | None = None,
        ignore_error: bool = False,
    ) -> str: ...

    @overload
    async def get_text_content(
        self,
        repo_url: str,
        file_path: list[str],
        branch: str = "main",
        repo_type: RepoType | None = None,
        ignore_error: bool = False,
    ) -> list[tuple[str, str]]: ...

    async def get_text_content(
        self,
        repo_url: str,
        file_path: str | list[str],
        branch: str = "main",
        repo_type: RepoType | None = None,
        ignore_error: bool = False,
    ) -> str | list[tuple[str, str]]:
        """
        获取仓库文本文件内容

        参数:
            repo_url: 仓库URL
            file_path: 文件路径
            branch: 分支名称
            repo_type: 仓库类型，如果为None则自动判断
            ignore_error: 是否忽略错误

        返回:
            str: 文件内容
        """
        # 确定仓库类型
        repo_name = (
            repo_url.split("/tree/")[0].split("/")[-1].replace(".git", "").strip()
        )
        if repo_type is None:
            try:
                return await self.get_aliyun_text_content(
                    repo_name, file_path, branch, ignore_error
                )
            except Exception:
                return await self.get_github_text_content(
                    repo_url, file_path, branch, ignore_error
                )

        try:
            if repo_type == RepoType.GITHUB:
                return await self.get_github_text_content(
                    repo_url, file_path, branch, ignore_error
                )

            elif repo_type == RepoType.ALIYUN:
                return await self.get_aliyun_text_content(
                    repo_name, file_path, branch, ignore_error
                )

        except Exception as e:
            if isinstance(e, FileNotFoundError | NetworkError | RepoManagerError):
                raise
            raise RepoManagerError(f"获取文件内容失败: {e}")

    async def list_directory_files(
        self,
        repo_url: str,
        directory_path: str = "",
        branch: str = "main",
        repo_type: RepoType | None = None,
        recursive: bool = True,
    ) -> list[RepoFileInfo]:
        """
        获取仓库目录下的所有文件路径。

        参数:
            repo_url: 仓库URL，可以包含 /tree/<branch>，会自动解析出分支和仓库地址。
            directory_path: 目录路径，默认为仓库根目录
            branch: 分支名称（若 repo_url 中包含 /tree/<branch>，则以 URL 中的为准）
            repo_type: 仓库类型，如果为None则自动判断
            recursive: 是否递归获取子目录文件

        返回:
            list[RepoFileInfo]: 文件信息列表
        """
        base_url = repo_url
        if "/tree/" in repo_url:
            base_url, tree_part = repo_url.split("/tree/", maxsplit=1)
            if tree_branch := tree_part.split("/", maxsplit=1)[0].strip():
                branch = tree_branch

        repo_name = base_url.split("/")[-1].replace(".git", "").strip()

        try:
            if repo_type is None:
                # 尝试阿里云，失败则尝试 GitHub
                try:
                    return await self._list_aliyun_directory_files(
                        repo_name, directory_path, branch, recursive
                    )
                except Exception as e:
                    logger.warning(
                        "获取阿里云目录文件失败，尝试GitHub", LOG_COMMAND, e=e
                    )
                    return await self._list_github_directory_files(
                        base_url, directory_path, branch, recursive
                    )
            if repo_type == RepoType.GITHUB:
                return await self._list_github_directory_files(
                    base_url, directory_path, branch, recursive
                )
            elif repo_type == RepoType.ALIYUN:
                return await self._list_aliyun_directory_files(
                    repo_name, directory_path, branch, recursive
                )
        except Exception as e:
            logger.error(f"获取目录文件列表失败: {directory_path}", LOG_COMMAND, e=e)
            if isinstance(e, FileNotFoundError | NetworkError | RepoManagerError):
                raise
            raise RepoManagerError(f"获取目录文件列表失败: {e}") from e

    async def _list_github_directory_files(
        self,
        repo_url: str,
        directory_path: str = "",
        branch: str = "main",
        recursive: bool = True,
        build_tree: bool = False,
    ) -> list[RepoFileInfo]:
        """
        获取GitHub仓库目录下的所有文件路径

        参数:
            repo_url: 仓库URL
            directory_path: 目录路径，默认为仓库根目录
            branch: 分支名称
            recursive: 是否递归获取子目录文件
            build_tree: 是否构建目录树

        返回:
            list[RepoFileInfo]: 文件信息列表
        """
        try:
            repo_info = GithubUtils.parse_github_url(repo_url)
            repo_info.branch = branch
            if await repo_info.update_repo_commit():
                logger.info(f"获取最新提交: {repo_info.branch}", LOG_COMMAND)
            else:
                logger.warning(f"获取最新提交失败: {repo_info}", LOG_COMMAND)

            # 获取仓库树信息
            strategy = GitHubStrategy()
            strategy.body = await strategy.parse_repo_info(repo_info)

            # 处理目录路径，确保格式正确
            if directory_path and not directory_path.endswith("/") and recursive:
                directory_path = f"{directory_path}/"

            # 获取文件列表
            file_list = []
            for tree_item in strategy.body.tree:
                # 如果不是递归模式，只获取当前目录下的文件
                if not recursive and "/" in tree_item.path.replace(
                    directory_path, "", 1
                ):
                    continue

                # 检查是否在指定目录下
                if directory_path and not tree_item.path.startswith(directory_path):
                    continue

                # 创建文件信息对象
                file_info = RepoFileInfo(
                    path=tree_item.path,
                    is_dir=tree_item.type == TreeType.DIR,
                    size=tree_item.size,
                    last_modified=None,  # GitHub API不直接提供最后修改时间
                )
                file_list.append(file_info)

            # 构建目录树结构
            if recursive and build_tree:
                file_list = self._build_directory_tree(file_list)

            return file_list

        except Exception as e:
            logger.error(
                f"获取GitHub目录文件列表失败: {directory_path}", LOG_COMMAND, e=e
            )
            raise

    async def _list_aliyun_directory_files(
        self,
        repo_name: str,
        directory_path: str = "",
        branch: str = "main",
        recursive: bool = True,
        build_tree: bool = False,
    ) -> list[RepoFileInfo]:
        """
        获取阿里云CodeUp仓库目录下的所有文件路径

        参数:
            repo_name: 仓库名称
            directory_path: 目录路径，默认为仓库根目录
            branch: 分支名称
            recursive: 是否递归获取子目录文件
            build_tree: 是否构建目录树

        返回:
            list[RepoFileInfo]: 文件信息列表
        """
        try:
            from zhenxun.utils.github_utils.models import AliyunFileInfo

            # 获取仓库树信息
            search_type = "RECURSIVE" if recursive else "DIRECT"
            tree_list = await AliyunFileInfo.get_repository_tree(
                repo=repo_name,
                path=directory_path,
                ref=branch,
                search_type=search_type,
            )

            # 创建文件信息对象列表
            file_list = []
            for tree_item in tree_list:
                file_info = RepoFileInfo(
                    path=tree_item.path,
                    is_dir=tree_item.type == AliyunTreeType.DIR,
                    size=None,  # 阿里云API不直接提供文件大小
                    last_modified=None,  # 阿里云API不直接提供最后修改时间
                )
                file_list.append(file_info)

            # 构建目录树结构
            if recursive and build_tree:
                file_list = self._build_directory_tree(file_list)

            return file_list

        except Exception as e:
            logger.error(
                f"获取阿里云目录文件列表失败: {directory_path}", LOG_COMMAND, e=e
            )
            raise

    def _build_directory_tree(
        self, file_list: list[RepoFileInfo]
    ) -> list[RepoFileInfo]:
        """
        构建目录树结构

        参数:
            file_list: 文件信息列表

        返回:
            list[RepoFileInfo]: 根目录下的文件信息列表
        """
        # 按路径排序，确保父目录在子目录之前
        file_list.sort(key=lambda x: x.path)
        # 创建路径到文件信息的映射
        path_map = {file_info.path: file_info for file_info in file_list}
        # 根目录文件列表
        root_files = []

        for file_info in file_list:
            if parent_path := "/".join(file_info.path.split("/")[:-1]):
                # 如果有父目录，将当前文件添加到父目录的子文件列表中
                if parent_path in path_map:
                    path_map[parent_path].children.append(file_info)
                else:
                    # 如果父目录不在列表中，创建一个虚拟的父目录
                    parent_info = RepoFileInfo(
                        path=parent_path, is_dir=True, children=[file_info]
                    )
                    path_map[parent_path] = parent_info
                    # 检查父目录的父目录
                    grand_parent_path = "/".join(parent_path.split("/")[:-1])
                    if grand_parent_path and grand_parent_path in path_map:
                        path_map[grand_parent_path].children.append(parent_info)
                    else:
                        root_files.append(parent_info)
            else:
                # 如果没有父目录，则是根目录下的文件
                root_files.append(file_info)

        # 返回根目录下的文件列表
        return [
            file
            for file in root_files
            if all(f.path != file.path for f in file_list if f != file)
        ]

    async def download_files(
        self,
        repo_url: str,
        file_path: tuple[str, Path] | list[tuple[str, Path]],
        branch: str = "main",
        repo_type: RepoType | None = None,
        ignore_error: bool = False,
    ) -> FileDownloadResult:
        """
        使用 Git 稀疏检出下载仓库中的文件或目录

        参数:
            repo_url: 仓库URL
            file_path: 仓库路径与本地目标路径的映射
            branch: 分支名称
            repo_type: 仓库类型，阿里云类型会自动转换 CodeUp 地址
            ignore_error: 是否忽略不存在的仓库路径

        返回:
            FileDownloadResult: 下载结果
        """
        file_paths = [file_path] if isinstance(file_path, tuple) else file_path
        if not file_paths:
            raise RepoManagerError("参数错误: file_path 不能为空")
        if any(not sparse_path.strip() for sparse_path, _ in file_paths):
            raise RepoManagerError("参数错误: 仓库路径不能为空")

        repo_name = (
            repo_url.split("/tree/")[0].split("/")[-1].replace(".git", "").strip()
        )
        result = FileDownloadResult(
            repo_type=repo_type,
            repo_name=repo_name,
            file_path=file_paths,
            version=branch,
        )
        return await self._handle_with_sparse_checkout(
            repo_url=repo_url,
            branch=branch,
            file_paths=file_paths,
            repo_type=repo_type,
            ignore_error=ignore_error,
            result=result,
        )

    async def _handle_with_sparse_checkout(
        self,
        repo_url: str,
        branch: str,
        file_paths: list[tuple[str, Path]],
        repo_type: RepoType | None,
        ignore_error: bool,
        result: FileDownloadResult,
    ) -> FileDownloadResult:
        try:
            clone_url = repo_url.split("/tree/", maxsplit=1)[0].rstrip("/")
            if not clone_url.endswith(".git"):
                clone_url += ".git"
            if repo_type == RepoType.ALIYUN:
                repo_name = clone_url.rsplit("/", maxsplit=1)[-1].removesuffix(".git")
                group_name = await get_aliyun_group_for_repo(repo_name)
                clone_url = prepare_aliyun_url(clone_url, group_name)

            file_path_mapping = dict(file_paths)
            with tempfile.TemporaryDirectory(
                prefix="repo_sparse_", dir=self.config.cache_dir
            ) as temp_dir:
                staging_dir = Path(temp_dir)
                downloaded_paths = await sparse_checkout_clone(
                    repo_url=clone_url,
                    branch=branch,
                    sparse_path=list(file_path_mapping),
                    target_dir=staging_dir,
                )
                missing_paths = set(file_path_mapping) - set(downloaded_paths)
                if missing_paths and not ignore_error:
                    missing = ", ".join(sorted(missing_paths))
                    raise RuntimeError(f"稀疏检出路径不存在: {missing}")

                for sparse_path in downloaded_paths:
                    source_path = staging_dir / sparse_path
                    target_path = file_path_mapping[sparse_path]
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                    shutil.move(str(source_path), str(target_path))

            total_size = 0
            for sparse_path in downloaded_paths:
                downloaded_path = file_path_mapping[sparse_path]
                if downloaded_path.is_file():
                    total_size += downloaded_path.stat().st_size
                elif downloaded_path.is_dir():
                    for file in downloaded_path.rglob("*"):
                        if file.is_file():
                            with contextlib.suppress(Exception):
                                total_size += file.stat().st_size
            result.success = True
            result.file_size = total_size
            logger.info(f"sparse-checkout 下载成功: {downloaded_paths}")
            return result
        except GitUnavailableError as e:
            logger.error(f"Git不可用: {e}")
            result.success = False
            result.error_message = (
                "下载仓库文件需要使用 Git，当前 Git 不可用，请安装 Git 后重试"
            )
            return result
        except Exception as e:
            logger.error(f"sparse-checkout 克隆失败: {e}")
            result.success = False
            result.error_message = str(e)
            return result
