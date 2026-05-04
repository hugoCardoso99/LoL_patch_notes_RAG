"""CLI entrypoint for the LoL Patch Notes RAG system."""

import logging
import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """League of Legends Patch Notes RAG System."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


# ─── Scrape ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--max-patches", "-n", type=int, default=None, help="Max patches to scrape")
@click.option("--delay", "-d", type=float, default=2.0, help="Delay between requests (seconds)")
def scrape(max_patches, delay):
    """Scrape League of Legends patch notes from the web."""
    from src.scraper import scrape_all

    console.print("[bold cyan]Starting patch notes scraper...[/bold cyan]")
    patches = scrape_all(max_patches=max_patches, delay=delay)
    console.print(f"[green]Scraped {len(patches)} patch notes[/green]")
    for p in patches:
        console.print(f"  • {p.patch_version}: {p.title} ({len(p.content)} chars)")


# ─── Ingest ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--max-patches", "-n", type=int, default=None, help="Max patches to ingest")
def ingest(max_patches):
    """Run the full ingestion pipeline (scrape → chunk → embed → store)."""
    from src.ingest import run_ingestion

    console.print("[bold cyan]Starting ingestion pipeline...[/bold cyan]")
    run_ingestion(max_patches=max_patches)
    console.print("[green]Ingestion complete![/green]")


# ─── Query ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", type=int, default=None, help="Number of chunks to retrieve")
@click.option("--patch", "-p", type=str, default=None, help="Filter to specific patch version")
def ask(question, top_k, patch):
    """Ask a question about League of Legends patch notes."""
    from src.rag import query

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with console.status("Thinking..."):
        result = query(question, top_k=top_k, patch_filter=patch)

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print(f"\n[dim]Sources: {', '.join(result.source_patches)}[/dim]")
    console.print(
        f"[dim]Retrieval: {result.retrieval_latency_seconds*1000:.0f}ms | "
        f"Generation: {result.generation_latency_seconds*1000:.0f}ms | "
        f"Total: {result.total_latency_seconds*1000:.0f}ms[/dim]"
    )
    console.print(
        f"[dim]Top similarity: {result.retrieval_scores[0]:.4f}[/dim]"
        if result.retrieval_scores else ""
    )


# ─── Interactive ─────────────────────────────────────────────────────────────


@cli.command()
@click.option("--top-k", "-k", type=int, default=None, help="Number of chunks to retrieve")
def chat(top_k):
    """Interactive chat mode for asking questions."""
    from src.rag import query

    console.print(Panel(
        "Ask questions about LoL patch notes.\n"
        "Type 'quit' or 'exit' to stop.",
        title="Interactive RAG Chat",
        border_style="cyan",
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("Thinking..."):
            result = query(question, top_k=top_k)

        console.print(f"\n[bold green]Assistant:[/bold green] {result.answer}")
        console.print(
            f"[dim](Sources: {', '.join(result.source_patches)} | "
            f"Latency: {result.total_latency_seconds*1000:.0f}ms)[/dim]"
        )


# ─── Evaluate ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--save/--no-save", default=True, help="Save report to disk")
def evaluate(save):
    """Run the evaluation pipeline (RAGAS + custom metrics)."""
    from src.evaluation import run_full_evaluation, print_report

    console.print("[bold cyan]Running RAG evaluation pipeline...[/bold cyan]")
    report = run_full_evaluation(save_report=save)
    print_report(report)


# ─── Status ──────────────────────────────────────────────────────────────────


@cli.command()
def status():
    """Show the current state of the RAG system (documents, chunks)."""
    from src.database import get_all_documents, get_document_chunk_count
    from rich.table import Table

    docs = get_all_documents()

    if not docs:
        console.print("[yellow]No documents ingested yet. Run 'ingest' first.[/yellow]")
        return

    table = Table(title=f"Ingested Documents ({len(docs)} total)")
    table.add_column("Patch", style="cyan")
    table.add_column("Title")
    table.add_column("Chunks", justify="right")
    table.add_column("Scraped At")

    for doc in docs:
        chunks = get_document_chunk_count(doc["id"])
        table.add_row(
            doc["patch_version"],
            doc["title"] or "-",
            str(chunks),
            str(doc["scraped_at"]),
        )

    console.print(table)


if __name__ == "__main__":
    cli()
