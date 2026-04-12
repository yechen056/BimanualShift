import logging

from omegaconf import DictConfig

from yarr.agents.agent import Agent


supported_agents = {
    "bimanual": ("BIMANUALSHIFT_PERACT",),
}


def create_agent(cfg: DictConfig) -> Agent:

    method_name = cfg.method.name
    agent_type = cfg.method.agent_type

    logging.info("Using method %s with type %s", method_name, agent_type)

    assert(method_name in supported_agents[agent_type])

    agent_fn = agent_fn_by_name(method_name)
    
    if agent_type == "bimanual":
        return agent_fn(cfg)
    raise Exception("invalid agent type")


def agent_fn_by_name(method_name: str) -> Agent:
    if method_name.startswith("BIMANUALSHIFT_PERACT"):
        from bift.methods import bimanualshift

        return bimanualshift.launch_utils.create_agent
    else:
        raise ValueError("Method %s does not exists." % method_name)
