"""RAG Lab CLI."""
import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Enterprise RAG Evaluation Lab CLI."""
    pass


@main.command()
def hello():
    """Quick health check."""
    click.echo("RAG Lab is ready.")
