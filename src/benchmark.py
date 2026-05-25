import asyncio
import aiohttp
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import asciichartpy as ascii_chart

@dataclass
class Node:
    name: str
    url: str
    status: str = "Unknown"
    models: List[str] = field(default_factory=list)

@dataclass
class BenchResult:
    user_id: int
    node: str
    duration: float
    eval_count: int
    tok_s: float
    status: str

class LLMFleetBenchmark:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.console = Console()
        self.results: List[BenchResult] = []
        self.start_time: float = 0
        self.end_time: float = 0

    async def check_node_health(self, node: Node):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{node.url}/api/tags", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        node.status = "Online"
                        node.models = [m['name'] for m in data.get('models', [])]
                    else:
                        node.status = f"Error {resp.status}"
        except Exception as e:
            node.status = "Offline"

    async def run_single_benchmark(self, session: aiohttp.ClientSession, user_id: int, node: Node, model: str, prompt: str) -> BenchResult:
        start = time.time()
        try:
            async with session.post(f"{node.url}/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }, timeout=120) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    duration = time.time() - start
                    eval_count = data.get('eval_count', 0)
                    tok_s = eval_count / duration if duration > 0 else 0
                    return BenchResult(user_id, node.name, duration, eval_count, tok_s, "Success")
                else:
                    return BenchResult(user_id, node.name, 0, 0, 0, f"HTTP {resp.status}")
        except Exception as e:
            return BenchResult(user_id, node.name, 0, 0, 0, str(e))

    async def run_concurrent_benchmark(self, num_users: int, model: str, prompt: str):
        self.results = []
        self.start_time = time.time()
        
        # Simple load balancing: distribute users across online nodes
        online_nodes = [n for n in self.nodes if n.status == "Online"]
        if not online_nodes:
            self.console.print("[red]No online nodes found![/]")
            return

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_users):
                node = online_nodes[i % len(online_nodes)]
                tasks.append(self.run_single_benchmark(session, i, node, model, prompt))
            
            self.results = await asyncio.gather(*tasks)
        
        self.end_time = time.time()

    def generate_table(self) -> Table:
        table = Table(title=f"Benchmark Results ({len(self.results)} Concurrent Users)")
        table.add_column("User ID", justify="right", style="cyan")
        table.add_column("Node", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Tokens", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("tok/s", justify="right", style="bold yellow")

        for res in self.results:
            table.add_row(
                str(res.user_id),
                res.node,
                res.status,
                str(res.eval_count),
                f"{res.duration:.2f}",
                f"{res.tok_s:.2f}"
            )
        
        total_tokens = sum(r.eval_count for r in self.results)
        total_duration = self.end_time - self.start_time
        agg_tok_s = total_tokens / total_duration if total_duration > 0 else 0
        
        table.add_section()
        table.add_row("AGGREGATE", "", "", str(total_tokens), f"{total_duration:.2f}", f"{agg_tok_s:.2f}", style="bold green")
        
        return table

async def main():
    fleet_nodes = [
        Node("silvia", "http://localhost:11434"),
        Node("yuki", "http://192.168.1.6:11434"),
        Node("kokkoro", "http://192.168.1.5:11434"),
        Node("pecorine", "http://192.168.1.8:11434"),
        Node("kumo", "http://192.168.1.10:11435")
    ]
    
    bench = LLMFleetBenchmark(fleet_nodes)
    
    console = Console()
    with console.status("[bold green]Checking fleet health...") as status:
        await asyncio.gather(*(bench.check_node_health(n) for n in bench.nodes))
    
    # Display health table
    health_table = Table(title="Fleet Health Status")
    health_table.add_column("Node")
    health_table.add_column("URL")
    health_table.add_column("Status")
    health_table.add_column("Models")
    for n in bench.nodes:
        health_table.add_row(n.name, n.url, n.status, ", ".join(n.models[:3]) + ("..." if len(n.models) > 3 else ""))
    console.print(health_table)

    # Run benchmark for 4 users as a test
    model = "qwen3:8b" # Default small model for testing
    prompt = "Explain quantum computing in one paragraph."
    
    for concurrent_count in [1, 4, 8]:
        console.print(f"\n[bold yellow]Starting Benchmark: {concurrent_count} users...[/]")
        await bench.run_concurrent_benchmark(concurrent_count, model, prompt)
        console.print(bench.generate_table())

if __name__ == "__main__":
    asyncio.run(main())
