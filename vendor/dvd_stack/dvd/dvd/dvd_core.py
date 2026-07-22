import copy
import inspect
import json
from typing import Annotated as A
from typing import Literal as L

import dvd.config as config
from dvd.build_database import (clip_search_tool, frame_inspect_tool,
                                global_browse_tool, init_single_video_db)
from dvd.func_call_shema import as_json_schema
from dvd.func_call_shema import doc as D
from dvd.prompts import get_prompts
from dvd.utils import call_openai_model_with_tools
from concurrent.futures import ThreadPoolExecutor, as_completed

TOPK = 16

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer of the user's question, and finish the conversation."""
    raise StopException(answer)


class DVDCoreAgent:
    def __init__(self, video_db_path, video_caption_path, max_iterations):
        self.tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, finish]
        if config.LITE_MODE:
            self.tools.remove(frame_inspect_tool)
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self._apply_prompt_schema_overrides()
        self.video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
        self.max_iterations = max_iterations
        self.messages = self._construct_messages()

    def _apply_prompt_schema_overrides(self):
        """Overwrite tool description / parameter descriptions from PROMPTS."""
        p = get_prompts()
        desc_map = {
            "finish": (p.finish_desc, {"answer": p.finish_answer_arg}),
            "frame_inspect_tool": (
                p.frame_inspect_desc,
                {
                    "database": p.frame_inspect_arg_database,
                    "question": p.frame_inspect_arg_question,
                    "time_ranges_hhmmss": p.frame_inspect_arg_time_ranges,
                },
            ),
            "clip_search_tool": (
                p.clip_search_desc,
                {
                    "database": p.clip_search_arg_database,
                    "event_description": p.clip_search_arg_event,
                    "top_k": p.clip_search_arg_topk,
                },
            ),
            "global_browse_tool": (
                p.global_browse_desc,
                {
                    "database": p.global_browse_arg_database,
                    "query": p.global_browse_arg_query,
                },
            ),
        }
        for schema in self.function_schemas:
            fn = schema["function"]
            override = desc_map.get(fn["name"])
            if not override:
                continue
            fn_desc, arg_descs = override
            fn["description"] = fn_desc
            props = fn.get("parameters", {}).get("properties", {})
            for arg_name, arg_desc in arg_descs.items():
                if arg_name in props:
                    props[arg_name]["description"] = arg_desc

    def _construct_messages(self):
        p = get_prompts()
        messages = [
            {"role": "system", "content": p.orchestrator_system},
            {"role": "user", "content": p.orchestrator_user_template},
        ]
        video_length = self.video_db.get_additional_data()['video_length']
        messages[-1]['content'] = messages[-1]['content'].replace("VIDEO_LENGTH", str(video_length))
        return messages

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _exec_tool(self, tool_call, msgs):
        name = tool_call["function"]["name"]
        if name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return

        # Parse arguments
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        if "database" in inspect.signature(self.name_to_function_map[name]).parameters:
            args["database"] = self.video_db
        
        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            raise

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, question) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration to avoid hanging
            if i == self.max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": get_prompts().force_finish_user,
                    }
                )

            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY,
            )
            if response is None:
                return None

            response.setdefault("role", "assistant")
            msgs.append(response)

            # Execute any requested tool calls
            try:
                for tool_call in (response.get("tool_calls") or []):
                    self._exec_tool(tool_call, msgs)
            except StopException:
                return msgs

        return msgs

    def parallel_run(self, questions, max_workers=4) -> list[list[dict]]:
        """
        Run multiple questions in parallel.
        """
        results = []
        results = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.run, q): idx
                for idx, q in enumerate(questions)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing question: {e}")
                    results[idx] = None
        return results

    # ------------------------------------------------------------------ #
    # Streaming (generator) loop
    # ------------------------------------------------------------------ #
    def stream_run(self, question):
        """
        A generator version of `run`.  
        Yields:
            dict: every assistant / tool message produced during reasoning.
        """
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration
            if i == self.max_iterations - 1:
                final_usr_msg = {
                    "role": "user",
                    "content": get_prompts().force_finish_user,
                }
                msgs.append(final_usr_msg)
                # Don't yield user messages to the UI

            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY,
            )
            if response is None:
                return

            response.setdefault("role", "assistant")
            msgs.append(response)
            yield response  # ← stream assistant reply

            # Execute any requested tool calls
            try:
                for tool_call in (response.get("tool_calls") or []):
                    # Yield a formatted message about the tool being called
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    yield {
                        "role": "tool_call",
                        "name": tool_name,
                        "arguments": tool_args
                    }
                    
                    self._exec_tool(tool_call, msgs)
                    # Only yield the tool result message
                    if msgs[-1].get("role") == "tool":
                        yield msgs[-1]  # ← stream tool observation
            except StopException:
                return


def single_run_wrapper(info) -> dict:
    qid, video_db_path, video_caption_path, question = info
    agent = DVDCoreAgent(video_db_path, video_caption_path, question)
    msgs = agent.run()
    return {qid: msgs}


def main():
    video_caption_path = "./video_database/i2qSxMVeVLI/captions/captions.json"
    video_db_path = "./video_database/i2qSxMVeVLI/database.json"
    question = """What color are the lights on the coaches' chairs when they turn around?
(A) Red
(B) White
(C) Black
(D) Yellow"""
    
    agent = DVDCoreAgent(video_db_path, video_caption_path, max_iterations=15)
    messages = agent.run(question)

    if messages:
        for m in messages:
            import pdb; pdb.set_trace()
            if m.get('role') == 'assistant':
                print(m)


if __name__ == "__main__":
    main()
