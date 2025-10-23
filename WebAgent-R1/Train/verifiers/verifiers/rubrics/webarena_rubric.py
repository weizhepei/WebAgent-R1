from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

from verifiers.envs.WebArena.test_webarena import evaluator_router

class WebArenaRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["thinking", ("answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            # self.llm_as_judge_reward_func,
            self.exact_answer_reward_func,
            # self.tool_execution_reward_func,
            # self.parser.get_format_reward_func(),
            # self.parser.get_xml_reward_func(),
        ]

    def exact_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""


        print(f'\n>>> completions: {completions}')
        print(f'\n>>> answer: {answer}')

        print(f'>>>\n length of pages: {len(kwargs["pages"])}, type: {type(kwargs["pages"][0])}')


        responses = [self.get_last_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]


    def llm_as_judge_reward_func(self, completions: List[List[Dict[str, str]]], pages: List[object], **kwargs) -> List[float]:

        scores = [0] * len(completions)
        for i, (page, completion) in enumerate(zip(pages, completions)):
            eval_types = kwargs["eval"][i]["eval_types"]
            evaluator = evaluator_router(config_file=None, eval_types=eval_types)

            last_action = parse_last_action(completion)

            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page # model has to return last pages in completions
            )  

            scores[i] = score 

        return scores

# case "string_match":
#     evaluators.append(StringEvaluator()) # pass last action
# case "url_match":
#     evaluators.append(URLExactEvaluator())
# case "program_html":
#     evaluators.append(HTMLContentExactEvaluator())
# case "page_image_query":