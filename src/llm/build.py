"""Build management: compile llama.cpp and install profile binaries."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llm.config import BuildProfile, load_config

app = typer.Typer(help="Build llama.cpp and manage build profiles.", no_args_is_help=True)
console = Console()

# Path to the llama.cpp submodule directory (relative to project root, resolved at runtime).
_SUBMODULE_DIR = Path("llama.cpp")


def _project_root() -> Path:
    """Return the project root (directory containing config.toml)."""
    from llm.config import find_config  # noqa: PLC0415

    return find_config().parent


def _submodule_path() -> Path:
    return _project_root() / _SUBMODULE_DIR


def _cmake_build_dir(profile: BuildProfile) -> Path:
    return _submodule_path() / profile.build_dir_name


def _assert_submodule() -> None:
    """Raise an error with helpful instructions if the submodule is not initialized."""
    sm = _submodule_path()
    if not sm.exists() or not (sm / "CMakeLists.txt").exists():
        console.print(
            f"[red]llama.cpp submodule not found at {sm}[/red]\n\n"
            "Initialize it with:\n"
            "  [bold]uv run llm build init[/bold]\n"
            "or:\n"
            "  [bold]git submodule update --init --recursive[/bold]"
        )
        raise typer.Exit(1)


def _run_build(profile: BuildProfile, install_path: Path, jobs: int, release: bool) -> None:
    """Configure, build, and install binaries for a single profile."""
    build_dir = _cmake_build_dir(profile)
    flags = profile.get_full_flags()
    build_type = "Release" if release else "Debug"

    console.print(f"\n[bold cyan]── Building profile: {profile.name} ──[/bold cyan]")
    console.print(f"  Flags : {' '.join(flags) if flags else '(none)'}")
    console.print(f"  Dir   : {build_dir}")

    # cmake configure
    cmake_cmd = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(_submodule_path()),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        *flags,
    ]
    console.print(f"\n[dim]$ {' '.join(cmake_cmd)}[/dim]")
    subprocess.run(cmake_cmd, check=True)

    # cmake build
    build_cmd = ["cmake", "--build", str(build_dir), "--config", build_type, f"-j{jobs}"]
    console.print(f"[dim]$ {' '.join(build_cmd)}[/dim]")
    subprocess.run(build_cmd, check=True)

    # Install binaries
    dest_dir = install_path / profile.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for binary in ("llama-server", "llama-bench"):
        # Look for binary in build/bin/ (standard cmake output location)
        candidates = [
            build_dir / "bin" / binary,
            build_dir / binary,
        ]
        src = next((p for p in candidates if p.exists()), None)
        if src:
            dst = dest_dir / binary
            shutil.copy2(src, dst)
            dst.chmod(dst.stat().st_mode | 0o111)
            console.print(f"  [green]Installed[/green] {binary} → {dst}")
        else:
            console.print(
                f"  [yellow]Warning:[/yellow] {binary} not found after build "
                f"(checked {candidates})"
            )

    console.print(f"  [green]✓[/green] Profile '{profile.name}' built and installed")


@app.command("init")
def init() -> None:
    """Initialize the llama.cpp git submodule.

    Runs ``git submodule update --init --recursive`` to clone llama.cpp
    into the ``llama.cpp/`` directory. Safe to run multiple times.

    Examples:

      uv run llm build init
    """
    cfg = load_config()
    sm = _submodule_path()
    if sm.exists() and (sm / "CMakeLists.txt").exists():
        # Already initialized — pull the configured commit
        console.print(f"[dim]Submodule already present at {sm}[/dim]")
        _checkout_commit(cfg.build.commit)
        return

    console.print("[bold]Initializing llama.cpp submodule...[/bold]")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        check=True,
        cwd=_project_root(),
    )
    _checkout_commit(cfg.build.commit)
    console.print("[green]✓[/green] Submodule initialized")


def _checkout_commit(commit: str) -> None:
    """Checkout the configured commit inside the submodule."""
    if commit == "HEAD":
        return  # stay on whatever the submodule points to
    sm = _submodule_path()
    console.print(f"  Checking out commit: [bold]{commit}[/bold]")
    subprocess.run(["git", "checkout", commit], check=True, cwd=sm)


@app.command("run")
def build_run(
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile name to build (default: active profile)."),
    ] = None,
    all_profiles: Annotated[
        bool,
        typer.Option("--all", "-a", help="Build all configured profiles."),
    ] = False,
) -> None:
    """Build llama.cpp with the active (or specified) profile and install binaries.

    Examples:

      uv run llm build run                     # build active profile

      uv run llm build run --profile vulkan-flash

      uv run llm build run --all               # build all profiles
    """
    _assert_submodule()
    cfg = load_config()
    bc = cfg.build

    if not bc.profiles:
        console.print(
            "[yellow]No build profiles configured.[/yellow]  "
            "Add [[build.profiles]] entries to config.toml."
        )
        raise typer.Exit(1)

    if all_profiles:
        profiles_to_build = bc.profiles
    elif profile:
        p = bc.get_profile(profile)
        if p is None:
            console.print(
                f"[red]Unknown profile:[/red] '{profile}'\n"
                f"Available: {', '.join(bc.profile_names())}"
            )
            raise typer.Exit(1)
        profiles_to_build = [p]
    else:
        active = bc.active_profile
        if active is None:
            console.print("[yellow]No active profile.[/yellow] Add [[build.profiles]] to config.toml.")
            raise typer.Exit(1)
        profiles_to_build = [active]

    jobs = bc.jobs_count()
    install_path = bc.install_path
    console.print(f"Install dir : {install_path}")
    console.print(f"Build jobs  : {jobs}")

    for p in profiles_to_build:
        _run_build(p, install_path, jobs, bc.release)

    console.print(f"\n[green]✓[/green] {len(profiles_to_build)} profile(s) built successfully.")


@app.command("update")
def update(
    commit: Annotated[
        str | None,
        typer.Argument(help="Git commit/branch/tag to update to (default: HEAD)."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile to rebuild after update (default: active)."),
    ] = None,
    all_profiles: Annotated[
        bool,
        typer.Option("--all", "-a", help="Rebuild all profiles after update."),
    ] = False,
) -> None:
    """Pull the latest llama.cpp and rebuild.

    Examples:

      uv run llm build update               # pull latest, rebuild active profile

      uv run llm build update abc1234       # pin to a specific commit

      uv run llm build update --all         # pull latest, rebuild all profiles
    """
    _assert_submodule()
    sm = _submodule_path()
    target = commit or "HEAD"

    console.print(f"[bold]Updating llama.cpp submodule → {target}[/bold]")
    subprocess.run(["git", "fetch", "origin"], check=True, cwd=sm)

    if target == "HEAD":
        subprocess.run(["git", "pull", "origin", "HEAD"], check=True, cwd=sm)
    else:
        subprocess.run(["git", "checkout", target], check=True, cwd=sm)

    actual = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=sm,
    ).stdout.strip()
    console.print(f"  Now at: [bold]{actual}[/bold]")

    # Rebuild
    build_run(profile=profile, all_profiles=all_profiles)


@app.command("clean")
def clean(
    profile: Annotated[
        str | None,
        typer.Option("--name", help="Clean a specific profile's build directory."),
    ] = None,
    all_profiles: Annotated[
        bool,
        typer.Option("--all", help="Clean all profile build directories."),
    ] = False,
) -> None:
    """Remove profile build directories to force a clean rebuild.

    Examples:

      uv run llm build clean --name vulkan-flash

      uv run llm build clean --all
    """
    cfg = load_config()
    bc = cfg.build

    if not all_profiles and not profile:
        console.print("Specify [bold]--name <profile>[/bold] or [bold]--all[/bold].")
        raise typer.Exit(1)

    profiles_to_clean = bc.profiles if all_profiles else (
        [bc.get_profile(profile)] if bc.get_profile(profile) else []
    )
    if not profiles_to_clean:
        console.print(f"[red]Profile not found:[/red] '{profile}'")
        raise typer.Exit(1)

    for p in profiles_to_clean:
        d = _cmake_build_dir(p)
        if d.exists():
            shutil.rmtree(d)
            console.print(f"  [green]Removed[/green] {d}")
        else:
            console.print(f"  [dim]Already clean:[/dim] {d}")

    console.print("[green]✓[/green] Clean complete.")


@app.command("info")
def info() -> None:
    """Show the current submodule commit, profiles, and installed binary paths.

    Examples:

      uv run llm build info
    """
    cfg = load_config()
    bc = cfg.build

    # Submodule commit
    sm = _submodule_path()
    if sm.exists() and (sm / "CMakeLists.txt").exists():
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=sm,
        )
        commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "[dim]unknown[/dim]"
    else:
        commit = "[yellow]not initialized[/yellow]"

    console.print("\n[bold]llama.cpp submodule[/bold]")
    console.print(f"  Path   : {sm}")
    console.print(f"  Commit : {commit}")
    console.print(f"  Config commit: {bc.commit}")

    if not bc.profiles:
        console.print("\n[yellow]No build profiles configured.[/yellow]")
        return

    install_path = bc.install_path
    t = Table(title="Build Profiles", show_header=True)
    t.add_column("Name", style="cyan")
    t.add_column("Backend")
    t.add_column("Flags", style="dim")
    t.add_column("Build dir")
    t.add_column("Installed", style="dim")

    for p in bc.profiles:
        is_active = p == bc.active_profile
        name_str = f"[bold]{p.name}[/bold] ★" if is_active else p.name
        flags_str = " ".join(p.get_full_flags()) or "(none)"
        build_exists = "[green]✓[/green]" if _cmake_build_dir(p).exists() else "[red]✗[/red]"
        installed_bin = p.installed_server_bin(install_path)
        installed_str = "[green]✓[/green]" if installed_bin.exists() else "[red]✗[/red]"
        t.add_row(
            name_str,
            p.backend or "(flags only)",
            flags_str,
            build_exists,
            installed_str,
        )

    console.print(t)
    console.print(f"\n  Install dir : {install_path}")
    console.print(f"  Build jobs  : {bc.jobs} ({bc.jobs_count()} cores)")
