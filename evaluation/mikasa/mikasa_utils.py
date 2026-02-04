import time
import warnings
warnings.filterwarnings('ignore', message='.*env\\.\\w+ to get variables from other wrappers is deprecated.*')

import gymnasium as gym
from gymnasium import spaces

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

import mikasa_robo_suite
from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *


class CameraWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        sensor_data_space = env.observation_space.spaces['state'].spaces['sensor_data']
        base_cam_space = sensor_data_space.spaces['base_camera'].spaces['rgb']
        hand_cam_space = sensor_data_space.spaces['hand_camera'].spaces['rgb']

        self.observation_space = spaces.Dict({
            'image_primary': base_cam_space,
            'image_wrist': hand_cam_space,
        })

    def observation(self, obs: dict) -> dict:
        new_obs = {
            'image_primary': obs['sensor_data']['base_camera']['rgb'],
            'image_wrist': obs['sensor_data']['hand_camera']['rgb'],
        }
        return new_obs


def get_mikasa_eval_env(args):

    TIME = time.strftime('%Y%m%d_%H%M%S')

    if args.env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'cup_with_ball_number'
        prompt_info = None
    elif args.env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                            'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                            'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'angle_diff'
        prompt_info = 'target_angle'
    elif args.env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}), # camera works only for t ~ [0, 19]
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    else:
        raise ValueError(f"Unknown environment: {args.env_id}")

    # print('\n' + '='*75)
    # print('║' + ' '*24 + 'Environment Configuration' + ' '*24 + '║')
    # print('='*75)
    # print('║' + f' Environment ID: {args.env_id}'.ljust(73) + '║')
    # print('║' + f' Oracle Info:    {oracle_info}'.ljust(73) + '║')
    # print('║ Wrappers:'.ljust(74) + '║')
    # for wrapper, kwargs in wrappers_list:
    #     print('║    ├─ ' + wrapper.__name__.ljust(65) + '║')
    #     if kwargs:
    #         print('║    │  └─ ' + str(kwargs).ljust(65) + '║')
    # print('║' + '-'*73 + '║')
    #
    # state_msg = 'state will be used' if args.include_state else 'state will not be used'
    # print('║' + f' include_state:       {str(args.include_state):<5} │ {state_msg}'.ljust(68) + '║')
    #
    # rgb_msg = 'rgb images will be used' if args.include_rgb else 'rgb images will not be used'
    # print('║' + f' include_rgb:         {str(args.include_rgb):<5} │ {rgb_msg}'.ljust(68) + '║')
    #
    # oracle_msg = 'oracle info will be used' if args.include_oracle else 'oracle info will not be used'
    # print('║' + f' include_oracle:      {str(args.include_oracle):<5} │ {oracle_msg}'.ljust(68) + '║')
    #
    # joints_msg = 'joints will be used' if args.include_joints else 'joints will not be used'
    # print('║' + f' include_joints:      {str(args.include_joints):<5} │ {joints_msg}'.ljust(68) + '║')
    # print('='*75 + '\n')

    assert any([args.include_state, args.include_rgb]), "At least one of include_state or include_rgb must be True."
    assert not (args.include_joints and not args.include_rgb), "include_joints can only be True when include_rgb is True"

    if args.include_state and not args.include_rgb and not args.include_oracle and not args.include_joints:
        MODE = 'state'
    elif args.include_state and args.include_rgb and not args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_rgb is not implemented and does not make sense, since any environment can be solved only by using state")
        MODE = 'state_rgb'
    elif args.include_state and not args.include_rgb and args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_oracle is not implemented and does not make sense, since the state already contains oracle information")
        MODE = 'state_oracle'
    elif args.include_state and args.include_rgb and args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_rgb_oracle is not implemented and does not make sense, since any environment can be solved only by using state")
        MODE = 'state_rgb_oracle'
    elif not args.include_state and args.include_rgb and not args.include_oracle and not args.include_joints:
        MODE = 'rgb'
    elif not args.include_state and args.include_rgb and args.include_oracle and not args.include_joints:
        MODE = 'rgb_oracle'
    elif not args.include_state and args.include_rgb and args.include_joints and args.include_oracle:
        MODE = 'rgb_joints_oracle' # TODO: check if this is correct
    elif not args.include_state and args.include_rgb and args.include_joints and not args.include_oracle:
        MODE = 'rgb_joints'
    else:
        raise NotImplementedError(f"Unknown mode: {args.include_state=} {args.include_rgb=} {args.include_oracle=} {args.include_joints=}")

    SAVE_DIR = f'checkpoints/ppo_memtasks/{MODE}/{args.reward_mode}/{args.env_id}'

    print(f'{MODE=}')
    print(f'{prompt_info=}')

    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {})) # obs=torch.tensor -> dict with keys: state: obs, prompt: prompt, oracle_info: oracle_info

    env_kwargs = dict(sensor_configs=dict()) 

    if args.camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height

    env_kwargs["sensor_configs"]["shader_pack"] = args.shader
    env = gym.make(
        args.env_id,
        obs_mode="rgb",
        num_envs=1,
        render_mode="all",
        control_mode= args.control_mode,
        reconfiguration_freq = 1,
        sim_backend = args.sim_backend,
        **env_kwargs
    )

    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)


    for wrapper_class, wrapper_kwargs in wrappers_list:
        env = wrapper_class(env, **wrapper_kwargs)

    if args.save_video:
        eval_output_dir = args.record_dir

        print(f"Saving eval videos to {eval_output_dir}")
        env = RecordEpisode(
            env,
            output_dir=eval_output_dir,
            save_trajectory=False,
            save_video=True,          
            save_video_trigger=None,  
            save_on_reset=True
        )

    env = CameraWrapper(env)

    return env
