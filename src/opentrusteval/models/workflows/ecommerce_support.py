"""
Ecommerce Customer Support Workflows for Multi-Agent Orchestration

This module provides reusable workflow functions for common ecommerce support scenarios.
Extend this file or add new files in the workflows/ directory for other domains/use-cases.
"""

from src.opentrusteval.models.workflows.logging import log_workflow_run

def get_order_status_workflow_input(customer_id, order_id, status, expected_delivery):
    """
    Prepare input for an order status workflow.
    """
    return {
        "customer_id": customer_id,
        "query": "Where is my order?",
        "context": {
            "order_id": order_id,
            "order_status": status,
            "expected_delivery": expected_delivery
        }
    }

def run_order_status_workflow(orchestrator, customer_id, order_id, status, expected_delivery):
    """
    Run the order status workflow using the provided orchestrator (e.g., LangGraph).
    """
    workflow_input = get_order_status_workflow_input(customer_id, order_id, status, expected_delivery)
    try:
        result = orchestrator.run_workflow(workflow_input)
        log_workflow_run("order_status", workflow_input, result)
        return result
    except Exception as e:
        log_workflow_run("order_status", workflow_input, None, str(e))
        raise

# Example: Returns/Refunds workflow (template)
def get_returns_workflow_input(customer_id, order_id, reason):
    return {
        "customer_id": customer_id,
        "query": f"I want to return my order because {reason}",
        "context": {
            "order_id": order_id,
            "return_reason": reason
        }
    }

def run_returns_workflow(orchestrator, customer_id, order_id, reason):
    workflow_input = get_returns_workflow_input(customer_id, order_id, reason)
    try:
        result = orchestrator.run_workflow(workflow_input)
        log_workflow_run("returns", workflow_input, result)
        return result
    except Exception as e:
        log_workflow_run("returns", workflow_input, None, str(e))
        raise

# Example: Escalation workflow (template)
def get_escalation_workflow_input(customer_id, issue):
    return {
        "customer_id": customer_id,
        "query": f"I need to speak to a human about: {issue}",
        "context": {
            "escalation_issue": issue
        }
    }

def run_escalation_workflow(orchestrator, customer_id, issue):
    workflow_input = get_escalation_workflow_input(customer_id, issue)
    try:
        result = orchestrator.run_workflow(workflow_input)
        log_workflow_run("escalation", workflow_input, result)
        return result
    except Exception as e:
        log_workflow_run("escalation", workflow_input, None, str(e))
        raise

# --- TEMPLATE FOR NEW WORKFLOWS ---
# def get_<workflow>_workflow_input(...):
#     ...
# def run_<workflow>_workflow(orchestrator, ...):
#     ... 