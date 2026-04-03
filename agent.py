from tools import run_task_with_llm_decision

def run_agent(task: str):
    if task:
        print("[Agent] Working...")
        output = run_task_with_llm_decision(task)
        print("\n[Agent] Output:")
        print(output)
    else:
        print("[Agent] No task provided.")