import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict

class PatchDifficultyEvaluator:
    def __init__(self, parser):
        self.parser = parser
        self.EMPTY = parser.tile_to_idx['-']
        self.GROUND = parser.tile_to_idx['X']
        self.BREAKABLE = parser.tile_to_idx['S']
        self.COIN = parser.tile_to_idx['o']
        self.ENEMY = parser.tile_to_idx['E']
        self.PIPE_LEFT = parser.tile_to_idx['[']
        self.PIPE_RIGHT = parser.tile_to_idx[']']
        self.QUESTION = parser.tile_to_idx['Q']
        self.BRICK_LEFT = parser.tile_to_idx['<']
        self.BRICK_RIGHT = parser.tile_to_idx['>']
        self.CANNON_BOTTOM = parser.tile_to_idx['b']
        self.CANNON_TOP = parser.tile_to_idx['B']

        self.MAX_ENEMY_DENSITY = 0.15
        self.MAX_CANNON_DENSITY = 0.15
        self.MAX_PIPE_DENSITY = 0.15
        self.MAX_JUMP_DENSITY = 0.2
        self.MAX_PLATFORM_DENSITY = 0.1

    def _count_enemies(self, patch):
        return np.sum(patch == self.ENEMY)

    def _count_cannons(self, patch):
        height, width = patch.shape
        cannon_count = 0

        for col in range(width):
            has_cannon_bottom = np.any(patch[:, col] == self.CANNON_BOTTOM)
            if has_cannon_bottom:
                cannon_count += 1

        return cannon_count

    def _count_pipes(self, patch):
        height, width = patch.shape
        pipe_count = 0

        for col in range(width):
            has_pipe_left = np.any(patch[:, col] == self.PIPE_LEFT)
            if has_pipe_left:
                pipe_count += 1

        return pipe_count

    def _count_jumps(self, patch):
        height, width = patch.shape
        jumps = 0

        ground_floor = patch[height - 1]
        gap_length = 0

        for col in range(width):
            if ground_floor[col] == self.EMPTY:
                gap_length += 1
            else:
                if gap_length > 0:
                    jumps += 1
                    if gap_length > 4:
                        jumps += (gap_length - 2) * 0.5
                gap_length = 0

        if gap_length > 0:
            jumps += 1
            if gap_length > 2:
                jumps += (gap_length - 2) * 0.5

        return int(jumps)

    def _count_elevated_platforms(self, patch):
            height, width = patch.shape
            platforms = 0

            PLATFORM_TILES = {self.GROUND, self.BREAKABLE, self.QUESTION}

            for row in range(height - 1):
                in_platform = False
                platform_start = -1

                for col in range(width):
                    if patch[row, col] in PLATFORM_TILES:
                        if not in_platform:
                            in_platform = True
                            platform_start = col
                    else:
                        if in_platform:
                            is_floating = False
                            for c in range(platform_start, col):
                                for r in range(row + 1, height):
                                    if patch[r, c] == self.EMPTY:
                                        is_floating = True
                                        break
                                if is_floating:
                                    break

                            if is_floating:
                                platforms += 1

                            in_platform = False

                if in_platform:
                    is_floating = False
                    for c in range(platform_start, width):
                        for r in range(row + 1, height):
                            if patch[r, c] == self.EMPTY:
                                is_floating = True
                                break
                        if is_floating:
                            break

                    if is_floating:
                        platforms += 1

            return platforms


    def evaluate_patch(self, patch, metadata=None):
        height, width = patch.shape
        total_tiles = height * width

        enemies = self._count_enemies(patch)
        cannons = self._count_cannons(patch)
        pipes = self._count_pipes(patch)
        jumps = self._count_jumps(patch)
        platforms = self._count_elevated_platforms(patch)

        enemy_density = enemies / width
        cannon_density = cannons / width
        pipe_density = pipes / width
        jump_density = jumps / width
        platform_density = platforms / width

        enemy_term = min(enemy_density / self.MAX_ENEMY_DENSITY, 1.0)
        cannon_term = min(cannon_density / self.MAX_CANNON_DENSITY, 1.0)
        jump_term = min(jump_density / self.MAX_JUMP_DENSITY, 1.0)
        pipe_term = min(pipe_density / self.MAX_PIPE_DENSITY, 1.0)
        platform_term = min(platform_density / self.MAX_PLATFORM_DENSITY, 1.0)

        raw_diff = (
            0.60 * enemy_term +
            0.30 * cannon_term +
            0.20 * jump_term +
            0.30 * platform_term +
            0.20 * pipe_term
        )

        diff_score = min(raw_diff , 1.0)


        result = {
            "metadata": metadata,
            "counts": {
                "enemies": enemies,
                "cannons": cannons,
                "pipes": pipes,
                "jumps": jumps,
                "platforms": platforms,
                "total_tiles": total_tiles
            },
            "densities": {
                "enemy_density": round(enemy_density, 3),
                "cannon_density": round(cannon_density, 3),
                "pipe_density": round(pipe_density, 3),
                "jump_density": round(jump_density, 3),
                "platform_density": round(platform_density, 3)
            },
            "scores": {
                "difficulty_score": round(diff_score, 3),
            },
        }

        return result

    def evaluate_patches_batch(self, patches, metadata_list=None):
        if metadata_list is None:
            metadata_list = [None] * len(patches)

        results = []
        for i in range(len(patches)):
            result = self.evaluate_patch(patches[i], metadata_list[i])
            results.append(result)

        return results