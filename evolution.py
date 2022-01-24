import copy
from random import choice, randint, seed

import numpy as np

from player import Player


def get_fittness(elem):
    return elem.fitness


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)
        copy_players = [self.clone_player(player) for player in players]
        copy_players.sort(key=get_fittness, reverse=True)
        # TODO (Additional: Learning curve)
        return copy_players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            print(str(max(prev_players, key=get_fittness).fitness) + '\n')
            # TODO ( Parent selection and child generation )
            new_players = []
            for iteration in range(num_players):
                par_a, par_b = self.select_parents(prev_players)
                child_a, child_b = self.generate_children(par_a, par_b)
                new_players.append(child_a)
                new_players.append(child_b)
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    @staticmethod
    def select_parents(prev_players):
        par_a = choice(prev_players)
        par_b = choice(prev_players)
        return par_a, par_b

    def generate_children(self, par_a, par_b):
        value = randint(0, 2)
        child_a = self.clone_player(par_a)
        child_b = self.clone_player(par_b)
        for i in range(value):
            num = i + 1
            params_a = {'W': par_b.nn.parameters['W' + str(num)],
                        'b': par_b.nn.parameters['b' + str(num)]}
            params_b = {'W': par_a.nn.parameters['W' + str(num)],
                        'b': par_a.nn.parameters['b' + str(num)]}
            child_a.nn.change_layer_parameters(new_layer_parameters=params_a, layer_num=num)
            child_b.nn.change_layer_parameters(new_layer_parameters=params_b, layer_num=num)
        child_a.fitness = 0
        child_b.fitness = 0
        self.mutate(child_a)
        self.mutate(child_b)
        return child_a, child_b

    @staticmethod
    def mutate(child):
        for i in range(1, 3):
            val = randint(0, 100)
            if val > 30:
                params = {'W': np.random.normal(size=(child.nn.layer_sizes[i], child.nn.layer_sizes[i - 1])),
                          'b': np.zeros((child.nn.layer_sizes[i], 1))}
                child.nn.change_layer_parameters(new_layer_parameters=params, layer_num=i)
