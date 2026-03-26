import argparse


def patch_spec(model_spec, eval_harness_content):
  pass

def main(args):
    model_spec = None
    with open(args.model_spec_file, "r") as f:
        model_spec = f.read()
    
    #todo-1: parse llm response to extract proof body and helper lemmas
    #todo-2: Do safety check on complete proof body and helper lemmas to ensure they are well-formed and do not contain any disallowed constructs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Insert loop isolation attribute and model specs")
    parser.add_argument("--model-spec-file", type=str, required=True, help="Path to the LLM response file containing model specs")
    parser.add_argument("--eval-harness-task-file", type=str, required=True, help="Path to the evaluation harness task file")
    args = parser.parse_args()
    main(args)