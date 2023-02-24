import importlib

from copy import deepcopy

import torch

default_config = {
    'checkpoint_path': 'submission.ckpt',
    'rows': 6,
    'columns': 7,
    'inarow': 4,
    'num_actions': 7,
    'batch_size': 128,
    'player_ids': [1, 2],
    'device': 'cpu',
    'default_reward': 0.0,
}

config_ppo6 = deepcopy(default_config)
config_ppo6.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo7_multichannel = deepcopy(default_config)
config_ppo7_multichannel.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo8 = deepcopy(default_config)
config_ppo8.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo9 = deepcopy(default_config)
config_ppo9.update({
    'num_features': 1024,
    'hidden_dims': [256],
})
config_ppo9_multichannel = deepcopy(default_config)
config_ppo9_multichannel.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo12 = deepcopy(default_config)
config_ppo12.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo18 = deepcopy(default_config)
config_ppo18.update({
    'num_features': 512,
    'hidden_dims': [256],
})
config_ppo27 = deepcopy(default_config)
config_ppo27.update({
    'num_features': 512,
    'hidden_dims': [256],
})
config_ppo28 = deepcopy(default_config)
config_ppo28.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo29 = deepcopy(default_config)
config_ppo29.update({
    'num_features': 512,
    'hidden_dims': [128],
})

def select_config_from_feature_model(feature_model_path):
    if feature_model_path.endswith('ppo6.py'):
        config = config_ppo6
    elif feature_model_path.endswith('ppo7_multichannel.py'):
        config = config_ppo7_multichannel
    elif feature_model_path.endswith('ppo8.py'):
        config = config_ppo8
    elif feature_model_path.endswith('ppo9.py'):
        config = config_ppo9
    elif feature_model_path.endswith('ppo9_multichannel.py'):
        config = config_ppo9_multichannel
    elif feature_model_path.endswith('ppo12.py'):
        config = config_ppo12
    elif feature_model_path.endswith('ppo18.py'):
        config = config_ppo18
    elif feature_model_path.endswith('ppo28.py'):
        config = config_ppo28
    elif feature_model_path.endswith('ppo29.py'):
        config = config_ppo29
    else:
        raise ValueError(f'there is no matching config for the feature model path {feature_model_path}, please check how the name ends, it should end with ppo6.py or something similar')

    return config

def load_module_from_source(source_path):
    spec = importlib.util.spec_from_file_location('model', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_actor_critic(feature_model_path, rl_model_path, config, checkpoint_path, create_critic=False):
    feature_module = load_module_from_source(feature_model_path)

    def feature_model_creation_func(config):
        return feature_module.Model(config)

    actor_module = load_module_from_source(rl_model_path)
    actor = actor_module.Actor(config, feature_model_creation_func)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.train(False)

    critic = None
    if create_critic:
        critic = actor_module.Critic(config, actor.state_features_model)
        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic.train(False)

    return actor, critic
