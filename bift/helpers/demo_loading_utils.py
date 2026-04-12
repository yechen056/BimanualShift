import logging
from typing import List
import numpy as np
from rlbench.demo import Demo
import omegaconf


def _is_stopped(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change


def _is_stopped_right(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].right.gripper_open
        and obs.gripper_open == demo[i - 1].right.gripper_open
        and demo[i - 2].right.gripper_open == demo[i - 1].right.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change


def _is_stopped_left(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].left.gripper_open
        and obs.gripper_open == demo[i - 1].left.gripper_open
        and demo[i - 2].left.gripper_open == demo[i - 1].left.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change

def _keypoint_discovery_dualarm1101(demo: Demo, warm_up=60, cool_down=0, include_last_frame=False, stopping_delta=0.1) -> List[int]:
    """
    Custom keypoint discovery for dual arm real-world data.
    Handles warm_up and cool_down periods.
    """
    episode_keypoints = []
    if len(demo) == 0:
        return episode_keypoints

    # Initialize previous gripper states based on the warm_up frame or 0
    start_idx = max(0, warm_up)
    if start_idx >= len(demo):
        start_idx = 0
    
    prev_gripper_open_right = demo[start_idx].right.gripper_open
    prev_gripper_open_left = demo[start_idx].left.gripper_open
    
    stopped_buffer = 0
    
    # Determine the range to iterate over
    end_idx = len(demo) - cool_down
    
    for i in range(start_idx, end_idx):
        obs = demo[i]
        
        # Check if stopped (low velocity)
        # Note: Real robot data might need a slightly higher delta tolerance
        right_stopped = np.allclose(obs.right.joint_velocities, 0, atol=stopping_delta)
        left_stopped = np.allclose(obs.left.joint_velocities, 0, atol=stopping_delta)
        
        stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        
        # Check for gripper state change
        # Ideally gripper_open is binary (0 or 1), but compare carefully
        right_state_changed = obs.right.gripper_open != prev_gripper_open_right
        left_state_changed = obs.left.gripper_open != prev_gripper_open_left
        state_changed = right_state_changed or left_state_changed
        
        # Determine if this is a keypoint
        # i > start_idx prevents adding the very first frame as a 'change'
        is_keypoint = False
        if i != start_idx and (state_changed or stopped):
             is_keypoint = True

        # Add keypoint with simple debouncing (prevent adding consecutive frames)
        if is_keypoint:
            if len(episode_keypoints) == 0 or (i - episode_keypoints[-1] > 5):
                episode_keypoints.append(i)

        # Update previous states
        prev_gripper_open_right = obs.right.gripper_open
        prev_gripper_open_left = obs.left.gripper_open

    # Optionally include the last frame
    if include_last_frame:
        last_frame_idx = end_idx - 1
        if len(episode_keypoints) == 0 or episode_keypoints[-1] != last_frame_idx:
            episode_keypoints.append(last_frame_idx)
            
    print(f"Found {len(episode_keypoints)} keypoints in range [{start_idx}, {end_idx}].")
    return episode_keypoints
# def _keypoint_discovery_bimanual(demo: Demo, stopping_delta=0.1) -> List[int]:
#     # stopping_delta=0.0005
#     episode_keypoints = []
#     right_prev_gripper_open = demo[0].right.gripper_open
#     left_prev_gripper_open = demo[0].left.gripper_open
#     stopped_buffer = 0
#     for i, obs in enumerate(demo._observations):
#         right_stopped = _is_stopped_right(demo, i, obs.right, stopping_delta)
#         left_stopped = _is_stopped_left(demo, i, obs.left, stopping_delta)
#         stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
#         stopped_buffer = 4 if stopped else stopped_buffer - 1
#         # if change in gripper, or end of episode.
#         last = i == (len(demo) - 1)
#         right_state_changed = obs.right.gripper_open != right_prev_gripper_open
#         left_state_changed = obs.left.gripper_open != left_prev_gripper_open
#         state_changed = right_state_changed or left_state_changed
#         if i != 0 and (state_changed or last or stopped):
#             episode_keypoints.append(i)

#         right_prev_gripper_open = obs.right.gripper_open
#         left_prev_gripper_open = obs.left.gripper_open
#     if (
#         len(episode_keypoints) > 1
#         and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
#     ):
#         episode_keypoints.pop(-2)
#     print("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
#     return episode_keypoints

def _keypoint_discovery_bimanual(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    right_prev_gripper_open = demo[0].right.gripper_open
    left_prev_gripper_open = demo[0].left.gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo._observations):
        right_stopped = _is_stopped_right(demo, i, obs.right, stopping_delta)
        left_stopped = _is_stopped_left(demo, i, obs.left, stopping_delta)
        stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        right_state_changed = obs.right.gripper_open != right_prev_gripper_open
        left_state_changed = obs.left.gripper_open != left_prev_gripper_open
        state_changed = right_state_changed or left_state_changed
        if i != 0 and (state_changed or last or stopped):
            episode_keypoints.append(i)

        right_prev_gripper_open = obs.right.gripper_open
        left_prev_gripper_open = obs.left.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    print("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def _keypoint_discovery_unimanual(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopping_delta)
        stopped = (stopped_buffer <= 0) and stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    print("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def _keypoint_discovery_heuristic(demo: Demo, stopping_delta=0.1) -> List[int]:
    if demo[0].is_bimanual:
        return _keypoint_discovery_bimanual(demo, stopping_delta)
    else:
        return _keypoint_discovery_unimanual(demo, stopping_delta)


def keypoint_discovery(demo: Demo, stopping_delta=0.1, method="heuristic") -> List[int]:
    episode_keypoints = []
    if method == "heuristic":
        return _keypoint_discovery_heuristic(demo, stopping_delta)

    elif method == "random":
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(range(len(demo)), size=20, replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == "fixed_interval":
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints
    elif isinstance(method, omegaconf.listconfig.ListConfig):
        return list(method)
    else:
        raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum
