# Third-Party System Integrations

This folder contains integration endpoints, adapters, and utilities for connecting OpenTrustEval to external chatbot, LLM, or business systems. All answers from these systems can be verified in real time or batch mode using OpenTrustEval APIs and webhooks.

## Structure
- `endpoints/` — API endpoints for real-time and batch verification
- `webhooks/` — Webhook receivers for async verification
- `adapters/` — Client code and utilities for connecting to 3rd-party systems

## Usage
- Use the provided endpoints to verify answers from external systems before responding to end users.
- Use webhooks for async or event-driven verification workflows.
- Extend adapters for your specific chatbot, LLM, or business platform.
