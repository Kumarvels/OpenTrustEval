"""
Healthcare Support Workflows for Multi-Agent Orchestration
"""

def get_appointment_status_workflow_input(patient_id, appointment_id):
    return {
        "patient_id": patient_id,
        "query": "What is the status of my appointment?",
        "context": {
            "appointment_id": appointment_id
        }
    }

def run_appointment_status_workflow(orchestrator, patient_id, appointment_id):
    workflow_input = get_appointment_status_workflow_input(patient_id, appointment_id)
    return orchestrator.run_workflow(workflow_input)

def get_prescription_refill_workflow_input(patient_id, prescription_id):
    return {
        "patient_id": patient_id,
        "query": "Can I get a refill for my prescription?",
        "context": {
            "prescription_id": prescription_id
        }
    }

def run_prescription_refill_workflow(orchestrator, patient_id, prescription_id):
    workflow_input = get_prescription_refill_workflow_input(patient_id, prescription_id)
    return orchestrator.run_workflow(workflow_input) 