import asyncio
from pathlib import Path
import shutil

import pytest
from pytest_mock import MockerFixture


async def _run_git(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode(errors="replace")


def _plugin_info(*, ali_url: str | None = None):
    from zhenxun.builtin_plugins.plugin_store.models import StorePluginInfo
    from zhenxun.utils.enum import PluginType

    return StorePluginInfo(
        name="测试插件",
        module="demo",
        module_path="demo",
        description="",
        usage="",
        author="tester",
        version="1.0.0",
        plugin_type=PluginType.NORMAL,
        is_dir=True,
        github_url="https://github.com/example/demo",
        ali_url=ali_url,
    )


def test_source_order() -> None:
    from zhenxun.builtin_plugins.plugin_store.data_source import StoreManager
    from zhenxun.utils.repo_utils.models import RepoType

    assert StoreManager._get_source_order(None) == (
        RepoType.ALIYUN,
        RepoType.GITHUB,
    )
    assert StoreManager._get_source_order("ali") == (RepoType.ALIYUN,)
    assert StoreManager._get_source_order("git") == (RepoType.GITHUB,)


def test_repository_branch_is_resolved_per_source() -> None:
    from zhenxun.builtin_plugins.plugin_store.data_source import StoreManager
    from zhenxun.utils.repo_utils.models import RepoType

    plugin_info = _plugin_info()
    plugin_info.github_url = "https://github.com/example/demo/tree/master"

    assert (
        StoreManager._get_plugin_repository_branch(plugin_info, RepoType.ALIYUN, "main")
        == "main"
    )
    assert (
        StoreManager._get_plugin_repository_branch(plugin_info, RepoType.GITHUB, "main")
        == "master"
    )


@pytest.mark.parametrize("is_external", [False, True])
async def test_default_source_falls_back_to_github_for_all_plugins(
    mocker: MockerFixture,
    tmp_path: Path,
    is_external: bool,
) -> None:
    from zhenxun.builtin_plugins.plugin_store.data_source import StoreManager
    from zhenxun.utils.repo_utils.models import (
        FileDownloadResult,
        RepoFileInfo,
        RepoType,
    )

    mock_base_path = mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source.BASE_PATH",
        new=tmp_path / "zhenxun",
    )
    source_calls: list[RepoType] = []
    download_calls: list[tuple[RepoType, list[str]]] = []
    files = [
        RepoFileInfo(path="demo/__init__.py", is_dir=False),
        RepoFileInfo(path="demo/assets/icon.png", is_dir=False),
        RepoFileInfo(path="demo/requirements.txt", is_dir=False),
    ]

    async def list_directory_files(
        repo_url: str,
        directory_path: str,
        branch: str,
        repo_type: RepoType,
    ) -> list[RepoFileInfo]:
        source_calls.append(repo_type)
        if repo_type == RepoType.ALIYUN:
            raise RuntimeError("aliyun unavailable")
        return files

    async def download_files(
        repo_url: str,
        file_path: list[tuple[str, Path]],
        branch: str,
        repo_type: RepoType,
        ignore_error: bool = False,
    ) -> FileDownloadResult:
        download_calls.append((repo_type, [path for path, _ in file_path]))
        for source_path, destination_path in file_path:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.endswith(".png"):
                destination_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            else:
                destination_path.write_text(source_path, encoding="utf-8")
        return FileDownloadResult(
            repo_type=repo_type,
            repo_name="demo",
            file_path=file_path,
            version=branch,
            success=True,
        )

    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.list_directory_files",
        side_effect=list_directory_files,
    )
    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.download_files",
        side_effect=download_files,
    )
    install_requirement = mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "VirtualEnvPackageManager.install_requirement",
    )

    await StoreManager.install_plugin_with_repo(
        _plugin_info(),
        is_external=is_external,
    )

    assert source_calls == [RepoType.ALIYUN, RepoType.GITHUB]
    assert download_calls == [
        (
            RepoType.GITHUB,
            [
                "demo/__init__.py",
                "demo/assets/icon.png",
                "demo/requirements.txt",
            ],
        )
    ]
    assert (
        mock_base_path / "plugins" / "demo" / "assets" / "icon.png"
    ).read_bytes() == b"\x89PNG\r\n\x1a\n"
    install_requirement.assert_awaited_once()


async def test_zero_byte_aliyun_binary_falls_back_to_github(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    from zhenxun.builtin_plugins.plugin_store.data_source import StoreManager
    from zhenxun.utils.repo_utils.models import (
        FileDownloadResult,
        RepoFileInfo,
        RepoType,
    )

    mock_base_path = mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source.BASE_PATH",
        new=tmp_path / "zhenxun",
    )
    calls: list[RepoType] = []
    files = [
        RepoFileInfo(path="demo/__init__.py", is_dir=False),
        RepoFileInfo(path="demo/icon.png", is_dir=False),
    ]

    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.list_directory_files",
        return_value=files,
    )

    async def download_files(
        repo_url: str,
        file_path: list[tuple[str, Path]],
        branch: str,
        repo_type: RepoType,
        ignore_error: bool = False,
    ) -> FileDownloadResult:
        calls.append(repo_type)
        for source_path, destination_path in file_path:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.endswith(".png"):
                content = b"" if repo_type == RepoType.ALIYUN else b"image"
                destination_path.write_bytes(content)
            else:
                destination_path.write_text("", encoding="utf-8")
        return FileDownloadResult(
            repo_type=repo_type,
            repo_name="demo",
            file_path=file_path,
            version=branch,
            success=True,
        )

    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.download_files",
        side_effect=download_files,
    )
    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "VirtualEnvPackageManager.install_requirement",
    )

    await StoreManager.install_plugin_with_repo(_plugin_info(), is_external=True)

    assert calls[:2] == [RepoType.ALIYUN, RepoType.GITHUB]
    assert (mock_base_path / "plugins" / "demo" / "icon.png").read_bytes() == b"image"


async def test_forced_aliyun_does_not_fall_back(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    from zhenxun.builtin_plugins.plugin_store.data_source import StoreManager
    from zhenxun.builtin_plugins.plugin_store.exceptions import PluginStoreException
    from zhenxun.utils.repo_utils.models import RepoType

    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source.BASE_PATH",
        new=tmp_path / "zhenxun",
    )
    list_directory_files = mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.list_directory_files",
        side_effect=RuntimeError("aliyun unavailable"),
    )

    with pytest.raises(PluginStoreException, match="阿里云"):
        await StoreManager.install_plugin_with_repo(
            _plugin_info(),
            source="ali",
        )

    assert list_directory_files.await_count == 1
    await_args = list_directory_files.await_args
    assert await_args is not None
    assert await_args.kwargs["repo_type"] == RepoType.ALIYUN


async def test_root_plugin_uses_exact_sparse_paths(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    from zhenxun.builtin_plugins.plugin_store.data_source import StoreManager
    from zhenxun.utils.repo_utils.models import (
        FileDownloadResult,
        RepoFileInfo,
        RepoType,
    )

    mock_base_path = mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source.BASE_PATH",
        new=tmp_path / "zhenxun",
    )
    plugin_info = _plugin_info(
        ali_url="https://codeup.aliyun.com/organization/group/demo-mirror"
    )
    plugin_info.module_path = "."
    files = [
        RepoFileInfo(path="__init__.py", is_dir=False),
        RepoFileInfo(path="assets/icon.png", is_dir=False),
        RepoFileInfo(path="requirements.txt", is_dir=False),
    ]
    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.list_directory_files",
        return_value=files,
    )
    downloaded_paths: list[str] = []

    async def download_files(
        repo_url: str,
        file_path: list[tuple[str, Path]],
        branch: str,
        repo_type: RepoType,
        ignore_error: bool = False,
    ) -> FileDownloadResult:
        downloaded_paths.extend(path for path, _ in file_path)
        for source_path, destination_path in file_path:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            content = b"image" if source_path.endswith(".png") else b""
            destination_path.write_bytes(content)
        return FileDownloadResult(
            repo_type=repo_type,
            repo_name="demo-mirror",
            file_path=file_path,
            version=branch,
            success=True,
        )

    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "RepoFileManager.download_files",
        side_effect=download_files,
    )
    mocker.patch(
        "zhenxun.builtin_plugins.plugin_store.data_source."
        "VirtualEnvPackageManager.install_requirement",
    )

    await StoreManager.install_plugin_with_repo(plugin_info, source="ali")

    assert downloaded_paths == [
        "__init__.py",
        "assets/icon.png",
        "requirements.txt",
    ]
    assert (
        mock_base_path / "plugins" / "demo" / "assets" / "icon.png"
    ).read_bytes() == b"image"


async def test_repo_manager_sparse_checkout_preserves_exact_paths(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    from zhenxun.utils.repo_utils import RepoFileManager
    from zhenxun.utils.repo_utils.models import RepoType

    async def sparse_checkout_clone(
        repo_url: str,
        branch: str,
        sparse_path: list[str],
        target_dir: Path,
    ) -> list[str]:
        assert repo_url == "https://github.com/example/demo.git"
        assert branch == "master"
        assert sparse_path == ["demo/assets/icon.png"]
        source = target_dir / sparse_path[0]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"image")
        return sparse_path

    sparse_checkout = mocker.patch(
        "zhenxun.utils.repo_utils.file_manager.sparse_checkout_clone",
        side_effect=sparse_checkout_clone,
    )
    target = tmp_path / "target" / "icon.png"
    result = await RepoFileManager.download_files(
        "https://github.com/example/demo/tree/master",
        [("demo/assets/icon.png", target)],
        "master",
        repo_type=RepoType.GITHUB,
    )

    assert result.success
    assert target.read_bytes() == b"image"
    sparse_checkout.assert_awaited_once()


async def test_sparse_checkout_retries_git_fetch(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    from zhenxun.utils.repo_utils.utils import sparse_checkout_clone

    mocker.patch("zhenxun.utils.repo_utils.utils.check_git", return_value=True)
    sleep = mocker.patch("zhenxun.utils.repo_utils.utils.asyncio.sleep")
    fetch_attempts = 0
    fetch_timeouts: list[float | None] = []

    async def run_git_command(
        command: str | list[str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[bool, str, str]:
        nonlocal fetch_attempts
        if isinstance(command, list) and "fetch" in command:
            fetch_attempts += 1
            fetch_timeouts.append(timeout_seconds)
            if fetch_attempts < 3:
                return False, "", "connection reset"
        return True, "", ""

    mocker.patch(
        "zhenxun.utils.repo_utils.utils.run_git_command",
        side_effect=run_git_command,
    )

    downloaded = await sparse_checkout_clone(
        repo_url="https://github.com/example/demo",
        branch="main",
        sparse_path=["demo/__init__.py"],
        target_dir=tmp_path / "target",
    )

    assert downloaded == []
    assert fetch_attempts == 3
    assert fetch_timeouts == [60, 60, 60]
    assert sleep.await_count == 2


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
async def test_git_checkout_preserves_binary(tmp_path: Path) -> None:
    from zhenxun.utils.repo_utils.utils import sparse_checkout_clone

    source_repo = tmp_path / "source"
    source_repo.mkdir()
    await _run_git("init", "-b", "main", str(source_repo))
    await _run_git("-C", str(source_repo), "config", "user.name", "test")
    await _run_git("-C", str(source_repo), "config", "user.email", "test@example.com")
    binary_content = b"\x89PNG\r\n\x1a\n\x00\x01\xffbinary"
    (source_repo / "icon.png").write_bytes(binary_content)
    (source_repo / "__init__.py").write_text("", encoding="utf-8")
    await _run_git("-C", str(source_repo), "add", ".")
    await _run_git("-C", str(source_repo), "commit", "-m", "test")

    target_dir = tmp_path / "target"
    downloaded = await sparse_checkout_clone(
        repo_url=source_repo.as_uri(),
        branch="main",
        sparse_path=["icon.png", "__init__.py"],
        target_dir=target_dir,
    )

    assert downloaded == ["icon.png", "__init__.py"]
    assert (target_dir / "icon.png").read_bytes() == binary_content
    assert not (target_dir / ".git").exists()
