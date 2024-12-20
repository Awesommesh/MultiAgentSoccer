from enum import IntEnum
import logging
from typing import Optional, Dict
import uuid

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np


def get_simulation_agent_id(gym_agent_id: int) -> int:
    """
    Simulation uses IDs that alternate teams,
    but the wrappers groups them sequentially.
    We need to convert between the two.
    """
    # TODO: use the same indexing as in the simulation to avoid this
    mapping = {
        0: 0,
        1: 2,
        2: 4,
        3: 6,
        4: 1,
        5: 3,
        6: 5,
        7: 7
    }
    return mapping[gym_agent_id]


class EnvConfigurationChannel(SideChannel):
    class ConfigurationType(IntEnum):
        PLAYER_POSITION = 0
        PLAYER_VELOCITY = 1
        PLAYER_ROTATION = 2
        BALL_POSITION = 3
        BALL_VELOCITY = 4

    def __init__(self) -> None:
        super().__init__(uuid.UUID("3f07928c-2b0e-494a-810b-5f0bbb7aaeca"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        logging.info(f"Message received from simulator: {msg.read_string()}")

    def set_parameters(
        self,
        players_states: Optional[Dict[int, Dict[str, float]]] = None,
        ball_state: Optional[Dict[str, float]] = None,
    ) -> None:

        if players_states is not None:
            for agent_id in players_states:
                if "position" in players_states[agent_id]:
                    msg = OutgoingMessage()
                    msg.write_int32(self.ConfigurationType.PLAYER_POSITION)
                    msg.write_int32(get_simulation_agent_id(agent_id))
                    msg.write_float32_list(players_states[agent_id]["position"])
                    super().queue_message_to_send(msg)

                if "velocity" in players_states[agent_id]:
                    msg = OutgoingMessage()
                    msg.write_int32(self.ConfigurationType.PLAYER_VELOCITY)
                    msg.write_int32(get_simulation_agent_id(agent_id))
                    msg.write_float32_list(players_states[agent_id]["velocity"])
                    super().queue_message_to_send(msg)

                # degrees
                if "rotation_y" in players_states[agent_id]:
                    msg = OutgoingMessage()
                    msg.write_int32(self.ConfigurationType.PLAYER_ROTATION)
                    msg.write_int32(get_simulation_agent_id(agent_id))
                    msg.write_float32(players_states[agent_id]["rotation_y"])
                    super().queue_message_to_send(msg)

        if ball_state is not None:
            if "position" in ball_state:
                msg = OutgoingMessage()
                msg.write_int32(self.ConfigurationType.BALL_POSITION)
                msg.write_float32_list(ball_state["position"])
                super().queue_message_to_send(msg)

            if "velocity" in ball_state:
                msg = OutgoingMessage()
                msg.write_int32(self.ConfigurationType.BALL_VELOCITY)
                msg.write_float32_list(ball_state["velocity"])
                super().queue_message_to_send(msg)
