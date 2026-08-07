"""Benchmark & validation harness for the AI investment pipeline (V1.1).

A pure *consumer* of the application: it drives the existing agents, prompt
builder, LLM provider, and parser to measure their cost, and implements none of
their logic. Nothing in the application imports this package, so it can never
affect production behaviour.
"""
