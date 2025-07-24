"""
Finance Q&A Workflows for Multi-Agent Orchestration
"""

def get_account_balance_workflow_input(customer_id, account_id):
    return {
        "customer_id": customer_id,
        "query": "What is my account balance?",
        "context": {
            "account_id": account_id
        }
    }

def run_account_balance_workflow(orchestrator, customer_id, account_id):
    workflow_input = get_account_balance_workflow_input(customer_id, account_id)
    return orchestrator.run_workflow(workflow_input)

def get_transaction_history_workflow_input(customer_id, account_id, period="last month"):
    return {
        "customer_id": customer_id,
        "query": f"Show my transactions for {period}",
        "context": {
            "account_id": account_id,
            "period": period
        }
    }

def run_transaction_history_workflow(orchestrator, customer_id, account_id, period="last month"):
    workflow_input = get_transaction_history_workflow_input(customer_id, account_id, period)
    return orchestrator.run_workflow(workflow_input) 