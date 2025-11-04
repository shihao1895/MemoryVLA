from typing import Callable, Optional, Tuple

from dexbotic.data.dataset.transform.action import (ActionNormAnd2String,
                                                    AddAction, AddTrajectory,
                                                    DeltaAction)
from dexbotic.data.dataset.transform.common import (Pipeline, ToDict, ToList,
                                                    ToNumpy)
from dexbotic.data.dataset.transform.language import (AddPromptTemplate,
                                                      ReplaceAnswer,
                                                      defalut_prompt_template)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal


def get_default_action_process_func(statistic_mapping: dict = {'default': {'min': -1,
                                                                           'max': 1}},
                                    vocab_size: int = 255,
                                    predict_length: int = 1,
                                    delta: bool = False,
                                    trajectory_length: int = 1,
                                    trajectory_padding_model: str = 'last',
                                    prompt_template: Tuple[str,
                                                           Callable[[str],
                                                                    str]] = defalut_prompt_template,
                                    load_multimodal: bool = True,
                                    add_prompt_template: bool = True,
                                    normalize_action: bool = True,
                                    replace_with_default_answer: Optional[str] = None,
                                    string_format: Optional[str] = ' {value}',
                                    padding_action: Optional[bool] = False):
    default_action_process_func = Pipeline([ToDict(),
                                            ToNumpy(),
                                            AddAction(predict_length=predict_length)])
    if delta:
        default_action_process_func.add(DeltaAction())

    if trajectory_length > 1:
        default_action_process_func.add(
            AddTrajectory(
                trajectory_length=trajectory_length,
                padding_mode=trajectory_padding_model,
                padding_action=padding_action))

    if normalize_action:
        default_action_process_func.add(
            ActionNormAnd2String(
                statistic_mapping=statistic_mapping,
                vocab_size=vocab_size,
                string_format=string_format))

    if load_multimodal:
        default_action_process_func.add(LoadMultiModal())

    if add_prompt_template:
        default_action_process_func.add(
            AddPromptTemplate(
                prompt_template=prompt_template))

    if replace_with_default_answer:
        default_action_process_func.add(ReplaceAnswer(
            default_answer=replace_with_default_answer))

    default_action_process_func.add(ToList())

    return default_action_process_func
