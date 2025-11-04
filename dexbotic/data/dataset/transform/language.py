from typing import Callable, Union


defalut_prompt_template = '<image>\nWhat action should the robot take to {prompt}?'


class AddPromptTemplate:
    """Add the prompt template to `prompt` in the episode_data_dict.

       Have no effect if the `is_robot` is not in the episode_data_dict or is_robot is False.
    """

    def __init__(self,
                 prompt_template: Union[str, Callable[[str], str]
                                        ] = defalut_prompt_template,
                 ):
        """Args:
            prompt_template: Tuple[str, (str) -> str], the prompt template for the robot. Default: defalut_prompt_template
        """
        if isinstance(prompt_template, str):
            def prompt_template(x, t=prompt_template): return t.format(prompt=x)

        self.prompt_template = prompt_template

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        # assume all data in the episode_data_dict has the same value of `is_robot`
        if 'is_robot' in episode_data_dict and episode_data_dict['is_robot'][0]:
            episode_data_dict['prompt'] = [
                self.prompt_template(_) for _ in episode_data_dict['prompt']]
        return episode_data_dict


class ReplaceAnswer:
    """Replace the `answer` in the episode_data_dict with a default string
    """

    def __init__(self, default_answer: str = ' '):
        """Args:
            default_answer: str, the default answer to replace the original answer. Default: ' '
        """
        self.default_answer = default_answer

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if 'answer' in episode_data_dict:
            episode_data_dict['answer'] = [
                self.default_answer for _ in episode_data_dict['answer']]
        return episode_data_dict

class ToConversation:
    """Convert the prompt and answer to a conversation format
    """

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if 'conversations' in episode_data_dict:
            return episode_data_dict
        episode_data_dict['conversations'] = [{'from': 'human', 'value': episode_data_dict.pop('prompt', '')},
                                              {'from': 'gpt', 'value': episode_data_dict.pop('answer', '')}]
        return episode_data_dict
