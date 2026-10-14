"""macOS (darwin) minimal support: no hardware audio, no torch."""
import sys


def test_dsp_candidates_prefer_arm64_on_darwin(monkeypatch):
    from src.processors import native_backend
    monkeypatch.setattr(sys, "platform", "darwin")
    candidates = native_backend._library_candidates()
    assert candidates[0] == "female_dsp_arm64.dylib"
    assert "female_dsp_x64.dylib" in candidates


def test_dsp_candidates_stay_windows_first(monkeypatch):
    from src.processors import native_backend
    monkeypatch.setattr(sys, "platform", "win32")
    assert native_backend._library_candidates() == ("female_dsp_x64.dll",)


def test_missing_dsp_reports_platform_names(monkeypatch, tmp_path):
    from src.processors import native_backend
    monkeypatch.setattr(sys, "platform", "darwin")
    # Point the loader at an empty directory by shadowing the file check.
    monkeypatch.setattr(native_backend.Path, "is_file", lambda self: False)
    try:
        native_backend._load_library()
    except RuntimeError as error:
        assert "female_dsp_arm64.dylib" in str(error)
    else:
        raise AssertionError("expected RuntimeError for missing DSP")


def test_mac_package_shares_linux_tarball():
    from src.app_update import pick_asset
    release = {"tag_name": "v0.13.3", "assets": [
        {"name": "koeiro-v0.13.3-linux.tar.gz", "browser_download_url": "https://example/linux.tgz", "size": 1},
    ]}
    asset = pick_asset(release, platform="mac", kind="package")
    assert asset is not None and asset["name"].endswith("-linux.tar.gz")


def test_mac_frozen_has_no_auto_update():
    from src.app_update import pick_asset
    release = {"tag_name": "v0.13.3", "assets": []}
    assert pick_asset(release, platform="mac", kind="frozen") is None


def test_asset_name_mac():
    from tools.build_release_asset import asset_name
    assert asset_name("v0.13.3", "mac") == "koeiro-v0.13.3-mac.tar.gz"


def test_data_dir_uses_library_on_macos(monkeypatch, tmp_path):
    import src.runtime_paths as paths
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    target = paths.data_dir()
    assert str(target).endswith("Library/Application Support/Koeiro") or "Application Support" in str(target)


def test_setup_plan_includes_brew_and_pypi_torch_on_mac(monkeypatch):
    import tools.setup_release as setup
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable", sys.executable)
    steps = setup.plan_steps(".", with_models=False, with_lavasr=False, recreate=True)
    titles = [title for title, _ in steps]
    assert any("brew" in title for title in titles)
    torch_step = next(argv for title, argv in steps if "torch" in title)
    assert "--index-url" not in torch_step
