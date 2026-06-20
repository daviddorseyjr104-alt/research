"""
Africa Pension Watch — Research Intelligence Agent
Entry point for interactive chat, scanning, and report generation.

Usage:
  python main.py                          # Interactive chat (default)
  python main.py chat                     # Interactive chat
  python main.py scan [--priority high]   # Scan all sources
  python main.py ingest <url>             # Ingest a specific URL
  python main.py generate                 # Generate a research output (guided)
  python main.py stats                    # Knowledge base statistics
  python main.py process                  # Summarize unsummarized documents
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

import config

console = Console()


def _check_api_key():
    if not config.ANTHROPIC_API_KEY:
        console.print(
            "[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key."
        )
        sys.exit(1)


def cmd_chat(args):
    """Interactive chat with the Africa Pension Watch Research Agent."""
    _check_api_key()
    from src import knowledge_base as kb
    from src.agent import Agent

    kb.initialize()
    agent = Agent()

    console.print(Panel(
        "[bold yellow]Africa Pension Watch[/bold yellow]\n"
        "[dim]Research Intelligence Agent[/dim]\n\n"
        "Ask me about pension systems, regulations, investment rules, reform agendas,\n"
        "or ask me to generate policy briefs, articles, or comparative reports.\n\n"
        "[dim]Commands: /scan · /ingest <url> · /stats · /reset · /quit[/dim]",
        border_style="yellow",
        expand=False,
    ))

    stat = kb.stats()
    if stat["total_documents"] == 0:
        console.print(
            "[dim]Knowledge base is empty. Type [bold]/scan[/bold] to fetch research from sources, "
            "or ask me questions and I'll work from built-in jurisdiction data.[/dim]\n"
        )
    else:
        console.print(f"[dim]Knowledge base: {stat['total_documents']} documents[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input == "/quit" or user_input == "/exit":
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input == "/reset":
            agent.reset()
            console.print("[dim]Conversation reset.[/dim]")
            continue

        if user_input == "/stats":
            _show_stats()
            continue

        if user_input == "/scan":
            _run_scan(priority="high")
            continue

        if user_input.startswith("/ingest "):
            url = user_input[8:].strip()
            _run_ingest(url)
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]Thinking...[/dim]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            response = agent.chat(user_input)

        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold yellow]Africa Pension Watch Agent[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        console.print()


def cmd_scan(args):
    """Scan all or priority sources for new pension research."""
    _check_api_key()
    from src import knowledge_base as kb, web_scanner

    kb.initialize()
    priority = getattr(args, "priority", None)
    console.print(f"[bold]Scanning sources[/bold]" + (f" (priority: {priority})" if priority else "") + "...")

    results = web_scanner.scan_all_sources(priority_filter=priority)
    total = sum(results.values())
    console.print(f"\n[green]Scan complete.[/green] {total} document(s) ingested across {len(results)} source(s).")

    if total > 0:
        console.print("[dim]Running AI summarization on new documents...[/dim]")
        from src import document_processor
        processed = document_processor.process_unsummarized(max_docs=50)
        console.print(f"[green]Summarized {processed} document(s).[/green]")


def cmd_ingest(args):
    """Ingest a specific URL."""
    _check_api_key()
    from src import knowledge_base as kb
    kb.initialize()
    _run_ingest(args.url)


def _run_ingest(url: str):
    from src import web_scanner, document_processor

    console.print(f"[dim]Fetching: {url}[/dim]")
    with Progress(SpinnerColumn(), TextColumn("[dim]Fetching...[/dim]"), console=console, transient=True) as p:
        p.add_task("", total=None)
        result = web_scanner.ingest_url(url)

    if not result["success"]:
        console.print(f"[red]Failed:[/red] {result.get('error', 'Unknown error')}")
        return

    console.print(f"[green]Ingested:[/green] {result['title']} (ID: {result['doc_id']})")
    console.print(f"[dim]Content: {result['content_length']:,} characters · {'PDF' if result.get('is_pdf') else 'HTML'}[/dim]")

    if Confirm.ask("Run AI summarization now?", default=True):
        with Progress(SpinnerColumn(), TextColumn("[dim]Summarizing...[/dim]"), console=console, transient=True) as p:
            p.add_task("", total=None)
            enrichment = document_processor.summarize_document(result["doc_id"])
        if "error" not in enrichment:
            console.print(f"[green]Summary:[/green] {enrichment.get('summary', '')[:200]}...")
        else:
            console.print(f"[red]Summarization failed:[/red] {enrichment['error']}")


def _run_scan(priority: str = ""):
    from src import web_scanner, document_processor
    console.print("[dim]Scanning sources...[/dim]")
    results = web_scanner.scan_all_sources(priority_filter=priority or None)
    total = sum(results.values())
    console.print(f"[green]{total} document(s) ingested.[/green]")
    if total > 0:
        processed = document_processor.process_unsummarized(max_docs=30)
        console.print(f"[green]Summarized {processed} document(s).[/green]")


def cmd_generate(args):
    """Guided report generation."""
    _check_api_key()
    from src import knowledge_base as kb, report_generator

    kb.initialize()

    output_types = report_generator.list_output_types()
    console.print("\n[bold]Available output types:[/bold]")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("#", width=3)
    table.add_column("Type")
    table.add_column("Description")
    for i, ot in enumerate(output_types, 1):
        table.add_row(str(i), ot["label"], ot["description"])
    console.print(table)

    choice = Prompt.ask("Select output type (number)")
    try:
        ot = output_types[int(choice) - 1]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/red]")
        return

    topic = Prompt.ask("Topic or research question")
    countries_raw = Prompt.ask("Countries to focus on (comma-separated, or Enter for Africa-wide)", default="")
    countries = [c.strip() for c in countries_raw.split(",") if c.strip()] if countries_raw else []
    deep = Confirm.ask("Use Claude Opus for deeper analysis (slower)?", default=False)

    console.print(f"\n[dim]Generating {ot['label']}...[/dim]")
    with Progress(SpinnerColumn(), TextColumn("[dim]Writing...[/dim]"), console=console, transient=True) as p:
        p.add_task("", total=None)
        result = report_generator.generate(
            output_type=ot["id"],
            topic=topic,
            countries=countries,
            use_deep_model=deep,
        )

    console.print(f"\n[green]Generated:[/green] {result['title']}")
    console.print(f"[dim]Saved to: {result['saved_to']}[/dim]")
    console.print(f"[dim]Model: {result['model_used']}[/dim]\n")
    console.print(Panel(Markdown(result["content"][:2000] + "\n\n*[Output truncated — see saved file]*"), border_style="yellow"))


def cmd_stats(args):
    """Show knowledge base statistics."""
    from src import knowledge_base as kb
    kb.initialize()
    _show_stats()


def _show_stats():
    from src import knowledge_base as kb

    stat = kb.stats()
    console.print(f"\n[bold]Knowledge Base Statistics[/bold]")
    console.print(f"Total documents: [bold]{stat['total_documents']}[/bold]")
    console.print(f"Last scan: [dim]{stat['last_scan']}[/dim]\n")

    if stat["by_type"]:
        t = Table(title="By Document Type", show_header=True, header_style="bold")
        t.add_column("Type")
        t.add_column("Count", justify="right")
        for row in stat["by_type"]:
            t.add_row(row["doc_type"] or "unknown", str(row["n"]))
        console.print(t)

    if stat["by_jurisdiction"]:
        t2 = Table(title="By Jurisdiction (top 15)", show_header=True, header_style="bold")
        t2.add_column("Jurisdiction")
        t2.add_column("Count", justify="right")
        for row in stat["by_jurisdiction"]:
            t2.add_row(row["jurisdiction"] or "unknown", str(row["n"]))
        console.print(t2)


def cmd_process(args):
    """Summarize documents that haven't been processed yet."""
    _check_api_key()
    from src import knowledge_base as kb, document_processor
    kb.initialize()
    console.print("[dim]Processing unsummarized documents...[/dim]")
    count = document_processor.process_unsummarized(max_docs=50)
    console.print(f"[green]Processed {count} document(s).[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Africa Pension Watch Research Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="Interactive chat (default)")

    scan_p = subparsers.add_parser("scan", help="Scan sources for new research")
    scan_p.add_argument("--priority", choices=["high", "medium", "low"], help="Filter by priority")

    ingest_p = subparsers.add_parser("ingest", help="Ingest a specific URL")
    ingest_p.add_argument("url", help="URL to fetch and ingest")

    subparsers.add_parser("generate", help="Generate a research output (guided)")
    subparsers.add_parser("stats", help="Show knowledge base statistics")
    subparsers.add_parser("process", help="Summarize unprocessed documents")

    args = parser.parse_args()
    command = args.command or "chat"

    dispatch = {
        "chat": cmd_chat,
        "scan": cmd_scan,
        "ingest": cmd_ingest,
        "generate": cmd_generate,
        "stats": cmd_stats,
        "process": cmd_process,
    }

    from src import knowledge_base as kb
    kb.initialize()

    dispatch[command](args)


if __name__ == "__main__":
    main()
