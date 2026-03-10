import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def validate_args_and_show_help():
    parser = argparse.ArgumentParser(
        description="Batch process videos to remove AI watermarks (Sora, Runway, Pika, Kling, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all .mp4 files in input folder
    demark-world -i /path/to/input -o /path/to/output
    # Process all .mov files
    demark-world -i /path/to/input -o /path/to/output --pattern "*.mov"
    # Use E2FGVI_HQ model (time-consistent, slower)
    demark-world -i /path/to/input -o /path/to/output --model e2fgvi_hq
    # Use bf16 + torch compile for faster inference (CUDA only)
    demark-world -i /path/to/input -o /path/to/output --model e2fgvi_hq --bf16 --torch-compile
    # Suppress internal progress bars
    demark-world -i /path/to/input -o /path/to/output --quiet
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input folder containing video files",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output folder for cleaned videos",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.mp4",
        help="File pattern to match (default: *.mp4)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Run in quiet mode (suppress tqdm and most logs).",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lama",
        choices=["lama", "e2fgvi_hq"],
        help="Model to use for watermark removal (default: lama). Options: lama (fast, may flicker), e2fgvi_hq (time consistent, slower)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for YOLO watermark detection (default: 4)",
    )

    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for E2FGVI_HQ model (CUDA only, faster after first run)",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable BF16 inference for E2FGVI_HQ model (CUDA only, ~2x faster)",
    )

    args = parser.parse_args()

    input_folder = Path(args.input).expanduser().resolve()
    output_folder = Path(args.output).expanduser().resolve()

    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}", file=sys.stderr)
        sys.exit(1)

    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)

    return input_folder, output_folder, args


def main():
    input_folder, output_folder, args = validate_args_and_show_help()

    pattern = args.pattern

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich.text import Text as RichText

    from src.demark_world.core import DeMarkWorld
    from src.demark_world.schemas import CleanerType

    console = Console()

    class SpeedColumnImpl(ProgressColumn):
        """Custom column to display processing speed in it/s format"""

        def render(self, task):
            if "Overall Progress" in task.description:
                return RichText("", style="")

            speed = task.finished_speed or task.speed
            if speed is None:
                return RichText("-- it/s", style="progress.data.speed")
            return RichText(f"{speed:.2f} it/s", style="cyan")

    class BatchProcessorImpl:
        """Batch video processor with rich progress tracking"""

        def __init__(
            self,
            input_folder: Path,
            output_folder: Path,
            pattern: str = "*.mp4",
            cleaner_type: CleanerType = CleanerType.LAMA,
            detect_batch_size: int = 4,
            enable_torch_compile: bool = False,
            use_bf16: bool = False,
        ):
            self.input_folder = input_folder
            self.output_folder = output_folder
            self.pattern = pattern
            self.demark_world = DeMarkWorld(
                cleaner_type=cleaner_type,
                enable_torch_compile=enable_torch_compile,
                detect_batch_size=detect_batch_size,
                use_bf16=use_bf16,
            )
            self.console = console

            self.successful: List[str] = []
            self.failed: Dict[str, str] = {}

        def show_banner(self):
            banner_text = Text()
            banner_text.append("DeMark-World", style="bold cyan")
            banner_text.append(" - AI Watermark Remover", style="bold magenta")

            panel = Panel(
                banner_text,
                box=box.DOUBLE,
                border_style="bright_blue",
                padding=(1, 2),
            )
            console.print(panel)
            console.print()

        def find_videos(self) -> List[Path]:
            video_files = list(self.input_folder.glob(self.pattern))
            return sorted(video_files)

        def process_batch(self):
            self.show_banner()

            video_files = self.find_videos()

            if not video_files:
                console.print(
                    f"[bold red]No files matching '{self.pattern}' found in {self.input_folder}[/bold red]"
                )
                return

            config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            config_table.add_row("Input folder:", f"[cyan]{self.input_folder}[/cyan]")
            config_table.add_row("Output folder:", f"[green]{self.output_folder}[/green]")
            config_table.add_row("Pattern:", f"[yellow]{self.pattern}[/yellow]")
            config_table.add_row(
                "Videos found:", f"[bold magenta]{len(video_files)}[/bold magenta]"
            )
            console.print(config_table)
            console.print()

            self.output_folder.mkdir(parents=True, exist_ok=True)

            start_time = datetime.now()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                SpeedColumnImpl(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                batch_task = progress.add_task(
                    "[cyan]Overall Progress", total=len(video_files)
                )

                for idx, input_path in enumerate(video_files, 1):
                    output_path = self.output_folder / f"cleaned_{input_path.name}"

                    progress.update(
                        batch_task,
                        description=f"[cyan]Overall Progress ({idx}/{len(video_files)})",
                    )

                    console.print(
                        f"\n[bold blue][{idx}/{len(video_files)}][/bold blue] "
                        f"[yellow]{input_path.name}[/yellow]"
                    )

                    try:
                        video_task = progress.add_task(
                            "  [green]Processing video", total=100
                        )

                        last_progress = [0]

                        def progress_callback(prog: int):
                            if prog > last_progress[0]:
                                progress.update(
                                    video_task, advance=prog - last_progress[0]
                                )
                                last_progress[0] = prog

                        self.demark_world.run(
                            input_path, output_path, progress_callback, quiet=args.quiet
                        )

                        if last_progress[0] < 100:
                            progress.update(video_task, advance=100 - last_progress[0])

                        progress.remove_task(video_task)

                        self.successful.append(input_path.name)
                        console.print(
                            f"  [bold green]Completed:[/bold green] {output_path.name}"
                        )

                    except Exception as e:
                        progress.remove_task(video_task)
                        self.failed[input_path.name] = str(e)
                        console.print(f"  [bold red]Error:[/bold red] {e}")

                    progress.update(batch_task, advance=1)

            self._print_summary(start_time)

        def _print_summary(self, start_time: datetime):
            end_time = datetime.now()
            duration = end_time - start_time

            console.print()

            summary_table = Table(
                show_header=False, box=box.ROUNDED, border_style="cyan"
            )
            summary_table.add_column("Metric", style="bold")
            summary_table.add_column("Value")

            summary_table.add_row("Total Time", f"[yellow]{duration}[/yellow]")
            summary_table.add_row(
                "Successful", f"[bold green]{len(self.successful)}[/bold green]"
            )
            summary_table.add_row(
                "Failed", f"[bold red]{len(self.failed)}[/bold red]"
            )
            summary_table.add_row(
                "Total",
                f"[bold magenta]{len(self.successful) + len(self.failed)}[/bold magenta]",
            )

            total = len(self.successful) + len(self.failed)
            success_rate = (len(self.successful) / total * 100) if total > 0 else 0
            summary_table.add_row(
                "Success Rate", f"[bold cyan]{success_rate:.1f}%[/bold cyan]"
            )

            summary_panel = Panel(
                summary_table,
                title="[bold white]BATCH PROCESSING SUMMARY[/bold white]",
                border_style="bright_cyan",
                box=box.DOUBLE,
            )
            console.print(summary_panel)

            if self.successful:
                console.print()
                success_table = Table(
                    title="[bold green]Successfully Processed[/bold green]",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold green",
                )
                success_table.add_column("#", style="dim", width=4)
                success_table.add_column("Filename", style="green")

                for idx, filename in enumerate(self.successful, 1):
                    success_table.add_row(str(idx), filename)

                console.print(success_table)

            if self.failed:
                console.print()
                failed_table = Table(
                    title="[bold red]Failed to Process[/bold red]",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold red",
                )
                failed_table.add_column("#", style="dim", width=4)
                failed_table.add_column("Filename", style="red")
                failed_table.add_column("Error", style="dim")

                for idx, (filename, error) in enumerate(self.failed.items(), 1):
                    error_msg = error if len(error) < 60 else error[:57] + "..."
                    failed_table.add_row(str(idx), filename, error_msg)

                console.print(failed_table)

            console.print()
            if len(self.failed) == 0:
                console.print(
                    "[bold green]All videos processed successfully![/bold green]",
                    justify="center",
                )
            else:
                console.print(
                    "[bold yellow]Some videos failed to process. Check errors above.[/bold yellow]",
                    justify="center",
                )
            console.print()

    try:
        cleaner_type = (
            CleanerType.LAMA if args.model == "lama" else CleanerType.E2FGVI_HQ
        )
        processor = BatchProcessorImpl(
            input_folder,
            output_folder,
            pattern,
            cleaner_type,
            detect_batch_size=args.batch_size,
            enable_torch_compile=args.torch_compile,
            use_bf16=args.bf16,
        )
        processor.process_batch()
    except KeyboardInterrupt:
        console.print()
        console.print(
            "[bold yellow]Processing interrupted by user[/bold yellow]",
            justify="center",
        )
        sys.exit(130)
    except Exception as e:
        console.print()
        console.print(f"[bold red]Fatal error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
