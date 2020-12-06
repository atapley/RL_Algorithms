from bases import ModelBase, TrainerBase, AgentBase

# Registers for Models, Trainers, and Agents.
# Each registry acts as a factory to create a modular codebase

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, model_name):
        def inner_wrapper(wrapped_class: ModelBase):
            cls._registry[model_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(self, model_name:str):
        model = self._registry.get(model_name)
        if not model:
            raise ValueError(model_name)
        return model


class TrainerRegistry:
    _registry = {}

    @classmethod
    def register(cls, trainer_name):
        def inner_wrapper(wrapped_class: TrainerBase):
            cls._registry[trainer_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(self, trainer_name:str):
        model = self._registry.get(trainer_name)
        if not model:
            raise ValueError(trainer_name)
        return model


class AgentRegistry:
    _registry = {}

    @classmethod
    def register(cls, agent_name):
        def inner_wrapper(wrapped_class: AgentBase):
            cls._registry[agent_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(self, agent_name:str):
        model = self._registry.get(agent_name)
        if not model:
            raise ValueError(agent_name)
        return model