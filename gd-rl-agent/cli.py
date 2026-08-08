"""Typer CLI entry point for the GD RL agent.

Commands:
  calibrate  — find GD window, set capture region, measure latency
  train      — run DQN training loop (Phase 4)
  eval       — evaluate a saved policy (Phase 4)
  watch      — observe agent without training (Phase 4)
"""

import json
import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="gd-rl", add_completion=False)
console = Console()


@app.command()
def calibrate(
    config_path: Path = typer.Option(
        Path("configs/agent_config.json"),
        "--config",
        "-c",
        help="Where to save the calibrated config.",
    ),
    latency_samples: int = typer.Option(
        30, "--samples", "-n", help="Number of jump samples for latency measurement."
    ),
) -> None:
    """Locate the GD window, calibrate capture region, and measure latency.

    Steps:
      1. Find the Geometry Dash window.
      2. Set capture region to the client area.
      3. Measure raw capture FPS (no inference).
      4. Measure pydirectinput injection latency.
      5. Interactively calibrate player_right_x and ground_y from a screenshot.
      6. Save AgentConfig to disk.

    The end-to-end latency (capture → inference stub → input) is printed so
    you can verify it is within ~2 game frames at GD's ~60fps (≤33ms).
    """
    from capture.screen_capture import ScreenCapture
    from capture.window_utils import WindowInfo, bring_to_foreground, find_gd_window
    from configs.config import AgentConfig, CaptureRegion, PlayerCalibration
    from input.controller import Controller
    import pydirectinput

    console.rule("[bold cyan]Phase 1 — Calibration[/bold cyan]")

    # ── Step 1: find window ──────────────────────────────────────────────────
    console.print("Searching for Geometry Dash window…")
    try:
        win: WindowInfo = find_gd_window()
    except RuntimeError as e:
        console.print(f"[red]ERROR:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"  Found: HWND={win.hwnd}  "
        f"window=({win.left},{win.top},{win.width}×{win.height})  "
        f"client=({win.client_left},{win.client_top},{win.client_width}×{win.client_height})"
    )

    region = CaptureRegion(
        left=win.client_left,
        top=win.client_top,
        width=win.client_width,
        height=win.client_height,
    )

    # ── Step 2: capture FPS ──────────────────────────────────────────────────
    console.print("\nMeasuring capture FPS (60 frames)…")
    with ScreenCapture(region) as cap:
        fps = cap.measure_fps(60)
    console.print(f"  Capture FPS: [bold green]{fps:.1f}[/bold green]")

    frame_budget_ms = 1000.0 / fps
    console.print(f"  Frame budget: {frame_budget_ms:.1f} ms")

    # ── Step 3: injection latency ────────────────────────────────────────────
    console.print(f"\nMeasuring input injection latency ({latency_samples} samples)…")
    console.print(
        "  [yellow]GD window will receive Space keypresses — make sure the game "
        "is on the main menu or paused.[/yellow]"
    )
    typer.confirm("  Ready to send keypresses?", abort=True)

    if not bring_to_foreground(win.hwnd):
        console.print(
            "[yellow]Could not automatically focus GD window. "
            "Click on it now, then press Enter.[/yellow]"
        )
        input()
    time.sleep(0.3)

    ctrl = Controller(win)
    lat = ctrl.measure_injection_latency(latency_samples)

    table = Table(title="Input injection latency")
    for col in ("mean_ms", "min_ms", "max_ms", "std_ms"):
        table.add_column(col, justify="right")
    table.add_row(
        f"{lat['mean_ms']:.2f}",
        f"{lat['min_ms']:.2f}",
        f"{lat['max_ms']:.2f}",
        f"{lat['std_ms']:.2f}",
    )
    console.print(table)

    # ── Step 4: end-to-end pipeline latency (capture + stub + input) ─────────
    console.print("\nMeasuring end-to-end pipeline latency (capture→stub→input)…")
    console.print(
        "  Note: timing excludes the mandatory key-hold sleep so the result\n"
        "  reflects actual pipeline overhead (capture + inference + keyDown/keyUp),\n"
        "  not the fixed 50ms tap duration that is the same on every machine."
    )
    e2e_samples: list[float] = []
    with ScreenCapture(region) as cap:
        for _ in range(latency_samples):
            t0 = time.perf_counter()
            _frame = cap.grab()           # capture
            time.sleep(0.001)             # inference stub (~1ms placeholder)
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            e2e_samples.append((time.perf_counter() - t0) * 1000)
            # full inter-sample gap includes the key-hold we skipped above
            time.sleep(0.25)

    e2e_mean = float(np.mean(e2e_samples))
    e2e_max = float(np.max(e2e_samples))
    gd_frame_ms = 1000.0 / 60.0  # GD runs at ~60fps

    console.print(f"  E2E mean: [bold]{e2e_mean:.1f} ms[/bold]  max: {e2e_max:.1f} ms")
    console.print(
        f"  GD frame: {gd_frame_ms:.1f} ms  "
        f"(2-frame budget excluding key-hold: {2*gd_frame_ms:.1f} ms)"
    )

    if e2e_mean > 2 * gd_frame_ms:
        console.print(
            "[bold red]WARNING:[/bold red] Pipeline overhead exceeds 2 GD frames "
            f"({e2e_mean:.1f} ms > {2*gd_frame_ms:.1f} ms, key-hold excluded). "
            "Frame-perfect jumps may not be achievable on this machine. "
            "Consider reducing capture region or deferring to lower FPS training."
        )
    else:
        console.print("[green]Pipeline overhead is within 2-frame budget.[/green]")

    # ── Step 5: interactive pixel calibration ────────────────────────────────
    console.print("\n[bold]Interactive pixel calibration[/bold]")
    console.print(
        "  We need two pixel values from a live screenshot of GD in cube mode:\n"
        "  • player_right_x — the X pixel at the RIGHT EDGE of the cube sprite\n"
        "  • ground_y       — the Y pixel of the GROUND LINE (top of the floor tiles)"
    )
    console.print(
        "  Open the screenshot saved to [cyan]calibration_frame.png[/cyan] "
        "in any image viewer, hover over those points, and enter their coords below."
    )

    with ScreenCapture(region) as cap:
        if not bring_to_foreground(win.hwnd):
            console.print(
                "[yellow]Could not automatically focus GD window. "
                "Click on it now, then press Enter.[/yellow]"
            )
            input()
        time.sleep(0.5)
        frame = cap.grab()

    import cv2
    save_path = "calibration_frame.png"
    if not cv2.imwrite(save_path, frame):
        console.print(
            f"[bold red]ERROR:[/bold red] Could not write {save_path}. "
            "Check disk space and working directory permissions."
        )
        raise typer.Exit(1)
    console.print(f"  Screenshot saved → {save_path}")

    player_right_x: int = typer.prompt("  player_right_x (pixels from LEFT of capture region)", type=int)
    ground_y: int = typer.prompt("  ground_y (pixels from TOP of capture region)", type=int)
    guard_px: int = typer.prompt("  guard_px (particle buffer, default 12)", default=12, type=int)
    scan_depth: int = typer.prompt("  scan_depth (how far ahead to scan, default 300)", default=300, type=int)
    cube_height_px: int = typer.prompt("  cube_height_px (default 36)", default=36, type=int)

    player_cal = PlayerCalibration(
        player_right_x=player_right_x,
        ground_y=ground_y,
        # Top of cube when grounded = ground_y minus the full cube sprite height.
        # Used by is_grounded pixel check in Phase 2 state_extractor.
        grounded_check_y=ground_y - cube_height_px,
        guard_px=guard_px,
        scan_depth=scan_depth,
        cube_height_px=cube_height_px,
    )

    # ── Step 6: save config ──────────────────────────────────────────────────
    cfg = AgentConfig(capture_region=region, player=player_cal)
    cfg.config_path = config_path
    cfg.save()
    console.print(f"\n[bold green]Config saved → {config_path}[/bold green]")

    # Summary
    console.rule("Calibration summary")
    summary = {
        "capture_fps": round(fps, 1),
        "frame_budget_ms": round(frame_budget_ms, 1),
        "injection_latency_mean_ms": round(lat["mean_ms"], 2),
        "e2e_latency_mean_ms": round(e2e_mean, 1),
        "e2e_latency_max_ms": round(e2e_max, 1),
        "within_2_frame_budget": e2e_mean <= 2 * gd_frame_ms,
        "player_right_x": player_right_x,
        "ground_y": ground_y,
        "guard_px": guard_px,
        "scan_depth": scan_depth,
    }
    console.print_json(json.dumps(summary))


@app.command()
def debug(
    config_path: Path = typer.Option(
        Path("configs/agent_config.json"), "--config", "-c"
    ),
    output: Path = typer.Option(
        Path("debug_frame.png"), "--output", "-o", help="Where to save annotated frame."
    ),
    duration: int = typer.Option(
        10, "--duration", "-d", help="How many seconds to run live loop (0 = single frame)."
    ),
) -> None:
    """Show what the agent sees: overlays ROI, detected obstacles, and feature vector.

    Saves annotated frames to disk and prints the feature vector each second so
    you can verify state_extractor output is correct before training.
    """
    import cv2 as _cv2
    from capture.screen_capture import ScreenCapture
    from capture.window_utils import find_gd_window, bring_to_foreground
    from configs.config import AgentConfig
    from vision.state_extractor import StateExtractor
    from vision.obstacle_detector import ObstacleDetector
    import time as _time

    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}. Run calibrate first.[/red]")
        raise typer.Exit(1)

    cfg = AgentConfig.load(config_path)
    extractor = StateExtractor(cfg)
    cal = cfg.player

    try:
        win = find_gd_window()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    bring_to_foreground(win.hwnd)

    roi_x0 = cal.player_right_x + cal.guard_px
    roi_x1 = cal.player_right_x + cal.scan_depth
    roi_y0 = max(0, cal.ground_y - 4 * cal.cube_height_px)
    roi_y1 = cal.ground_y + 4
    grounded_x = cal.player_right_x - cal.cube_height_px // 2
    grounded_y = cal.ground_y - cal.cube_height_px

    def _annotate(frame: np.ndarray, vec: np.ndarray, dead: bool) -> np.ndarray:
        out = frame.copy()
        # ROI rectangle (green)
        _cv2.rectangle(out, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 1)
        # Ground line (cyan)
        _cv2.line(out, (0, cal.ground_y), (frame.shape[1], cal.ground_y), (255, 255, 0), 1)
        # Player right edge (blue)
        _cv2.line(out, (cal.player_right_x, 0), (cal.player_right_x, frame.shape[0]), (255, 100, 0), 1)
        # Grounded check pixel (magenta dot)
        _cv2.circle(out, (grounded_x, grounded_y), 4, (255, 0, 255), -1)
        # Feature vector overlay
        labels = ["dist_obs", "dist_pit", "obs_h", "grounded"]
        for i, (lbl, val) in enumerate(zip(labels, vec)):
            text = f"{lbl}={val:.3f}"
            _cv2.putText(out, text, (10, 20 + i * 18), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if dead:
            _cv2.putText(out, "DEAD", (10, 110), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # Mark detected obstacle distance as vertical line
        dist_obs = float(vec[0])
        if dist_obs < 1.0:
            obs_col = roi_x0 + int(dist_obs * (roi_x1 - roi_x0))
            _cv2.line(out, (obs_col, roi_y0), (obs_col, roi_y1), (0, 0, 255), 2)
        # Mark detected pit distance as vertical line
        dist_pit = float(vec[1])
        if dist_pit < 1.0:
            pit_col = roi_x0 + int(dist_pit * (roi_x1 - roi_x0))
            _cv2.line(out, (pit_col, cal.ground_y - 10), (pit_col, cal.ground_y + 10), (0, 165, 255), 2)
        return out

    with ScreenCapture(cfg.capture_region) as cap:
        if duration == 0:
            frame = cap.grab()
            vec = extractor.extract(frame)
            dead = extractor.is_dead(frame)
            annotated = _annotate(frame, vec, dead)
            _cv2.imwrite(str(output), annotated)
            console.print(f"Feature vector: {vec}")
            console.print(f"Dead: {dead}")
            console.print(f"Saved {output}")
        else:
            t_end = _time.perf_counter() + duration
            frame_count = 0
            while _time.perf_counter() < t_end:
                frame = cap.grab()
                vec = extractor.extract(frame)
                dead = extractor.is_dead(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    annotated = _annotate(frame, vec, dead)
                    _cv2.imwrite(str(output), annotated)
                    console.print(f"[{frame_count:4d}] vec={vec}  dead={dead}")
            console.print(f"Done. Last frame saved to {output}")


@app.command()
def train(
    config_path: Path = typer.Option(Path("configs/agent_config.json"), "--config", "-c"),
    episodes: int = typer.Option(500, "--episodes", "-n"),
    save_every: int = typer.Option(50, "--save-every"),
    resume: Path = typer.Option(None, "--resume", help="Path to checkpoint to resume from."),
    double: bool = typer.Option(None, "--double/--no-double", help="Override use_double_dqn."),
    dueling: bool = typer.Option(None, "--dueling/--no-dueling", help="Override use_dueling_dqn."),
    run_name: str = typer.Option(None, "--run-name", help="Subdirectory under log_dir for this run."),
) -> None:
    """Train the DQN agent against the real GD window.

    Variant flags override config values so you can compare:
      python cli.py train --no-double --no-dueling --run-name vanilla
      python cli.py train --double --no-dueling   --run-name double
      python cli.py train --no-double --dueling   --run-name dueling
    """
    from configs.config import AgentConfig
    from agent.trainer import Trainer

    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}. Run calibrate first.[/red]")
        raise typer.Exit(1)

    cfg = AgentConfig.load(config_path)
    if double is not None:
        cfg.hp.use_double_dqn = double
    if dueling is not None:
        cfg.hp.use_dueling_dqn = dueling
    if run_name is not None:
        cfg.log_dir = cfg.log_dir / run_name

    variant = f"double={cfg.hp.use_double_dqn} dueling={cfg.hp.use_dueling_dqn}"
    console.print(f"Variant: {variant}  log_dir={cfg.log_dir}")

    trainer = Trainer(cfg)
    if resume is not None:
        trainer.load(resume)

    console.print(f"Starting training: {episodes} episodes, config={config_path}")
    trainer.train(episodes, save_every)


@app.command()
def watch(
    config_path: Path = typer.Option(Path("configs/agent_config.json"), "--config", "-c"),
    checkpoint: Path = typer.Option(None, "--checkpoint", "-k", help="Checkpoint .pt file."),
    episodes: int = typer.Option(5, "--episodes", "-n"),
) -> None:
    """Watch the trained agent play without updating weights."""
    import torch
    from configs.config import AgentConfig
    from agent.dqn import QNetwork
    from env.gd_env import GDEnv

    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        raise typer.Exit(1)

    cfg = AgentConfig.load(config_path)
    obs_dim = 6 if cfg.hp.use_obstacle_lookahead else 4
    net = QNetwork(obs_dim, cfg.hp.hidden_dim, 2, cfg.hp.use_dueling_dqn)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        net.load_state_dict(ckpt["online_state_dict"])
        console.print(f"Loaded {checkpoint}")
    else:
        console.print("[yellow]No checkpoint provided — using random policy.[/yellow]")

    net.eval()

    with GDEnv(cfg) as env:
        for ep in range(1, episodes + 1):
            obs = env.reset()
            ep_reward = 0.0
            steps = 0
            while True:
                with torch.no_grad():
                    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = int(net(t).argmax(dim=-1).item())
                result = env.step(action)
                ep_reward += result.reward
                steps += 1
                obs = result.obs
                if result.terminal:
                    break
            console.print(f"ep={ep}  steps={steps}  reward={ep_reward:.3f}  success={result.success}")


@app.command()
def eval(
    config_path: Path = typer.Option(Path("configs/agent_config.json"), "--config", "-c"),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-k"),
    episodes: int = typer.Option(20, "--episodes", "-n"),
) -> None:
    """Evaluate a saved checkpoint and print mean reward / success rate."""
    import torch
    from configs.config import AgentConfig
    from agent.dqn import QNetwork
    from env.gd_env import GDEnv

    cfg = AgentConfig.load(config_path)
    obs_dim = 6 if cfg.hp.use_obstacle_lookahead else 4
    net = QNetwork(obs_dim, cfg.hp.hidden_dim, 2, cfg.hp.use_dueling_dqn)
    ckpt = torch.load(checkpoint, map_location="cpu")
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()

    rewards, successes = [], []
    with GDEnv(cfg) as env:
        for _ in range(episodes):
            obs = env.reset()
            ep_r = 0.0
            while True:
                with torch.no_grad():
                    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = int(net(t).argmax(dim=-1).item())
                result = env.step(action)
                ep_r += result.reward
                obs = result.obs
                if result.terminal:
                    break
            rewards.append(ep_r)
            successes.append(result.success)

    console.print(f"Episodes: {episodes}")
    console.print(f"Mean reward:   {np.mean(rewards):.3f}")
    console.print(f"Success rate:  {np.mean(successes)*100:.1f}%")


if __name__ == "__main__":
    import pydirectinput  # noqa: F401 — ensure available before app starts
    app()
