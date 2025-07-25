import time

def run_workflow_chain(orchestrator, workflow_funcs, *args_list):
    results = []
    for func, args in zip(workflow_funcs, args_list):
        start = time.time()
        try:
            result = func(orchestrator, *args)
            results.append({
                "workflow": func.__name__,
                "input": args,
                "output": result,
                "error": None,
                "timestamp": start
            })
        except Exception as e:
            results.append({
                "workflow": func.__name__,
                "input": args,
                "output": None,
                "error": str(e),
                "timestamp": start
            })
            break  # Stop chain on error
    return results 