import copy
import random
from random import choice, randint, seed

import numpy as np

from player import Player


def get_fittness(elem):
    return elem.fitness


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def roulette_wheel(self, items, num_items):
        probabilities = self.get_probability_list(items)
        chosen = []
        for n in range(num_items):
            r = random.random()
            for (i, individual) in enumerate(items):
                if r <= probabilities[i]:
                    chosen.append(individual)
                    break
        return chosen

    def sus(self, items, num_items):
        points = self.generate_points(items, num_items)
        chosen = []
        while len(chosen) < num_items:
            random.shuffle(items)
            i = 0
            while i < len(points) and len(chosen) < num_items:
                j = 0
                sm = 0
                while j < len(items):
                    sm += items[j].fitness
                    if sm > points[i]:
                        chosen.append(items[j])
                        break
                    j += 1
                i += 1
        return chosen

    def top_k(self, items, num_items, k=2):
        chosen = []
        for iteration in range(num_items):
            best = None
            for i in range(k):
                r = randint(0, len(items) - 1)
                if best is None or (items[r].fitness > items[best].fitness):
                    best = r
            chosen.append(items[best])
        return chosen

    @staticmethod
    def generate_points(items, num_items):
        total_fitness = float(sum([item.fitness for item in items]))
        point_distance = total_fitness / num_items
        start_point = random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(num_items)]
        return points

    def next_population_selection(self, players, num_players, type_of_selection='sort'):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param type_of_selection: type of selection(top-k, roulette wheel, SUS, sort)
        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)
        copy_players = [self.clone_player(player) for player in players]
        if type_of_selection == 'sort':
            copy_players.sort(key=get_fittness, reverse=True)
            return copy_players[: num_players]
        elif type_of_selection == 'roulette wheel':
            return self.roulette_wheel(copy_players, num_players)
        elif type_of_selection == 'SUS':
            return self.sus(copy_players, num_players)
        elif type_of_selection == 'top-k':
            return self.top_k(copy_players, num_items=num_players, k=10)
        # TODO (Additional: Learning curve)

    def generate_new_population(self, num_players, prev_players=None, type_of_selection='random'):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param type_of_selection: type of parent selection(top-k, roulette wheel, SUS, random)
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
                par_a, par_b = self.select_parents(prev_players, type_of_selection=type_of_selection)
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

    def select_parents(self, prev_players, type_of_selection='random'):
        par_a = choice(prev_players)
        par_b = choice(prev_players)
        if type_of_selection == 'roulette wheel':
            par_a, par_b = self.roulette_wheel(prev_players, 2)
        elif type_of_selection == 'SUS':
            par_a, par_b = self.sus(prev_players, 2)
        elif type_of_selection == 'top-k':
            par_a, par_b = self.top_k(prev_players, 2, k=10)
        return par_a, par_b

    def generate_children(self, par_a, par_b):
        value = randint(0, len(par_a.la))
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
        for i in range(1, len(child.layer_sizes)):
            val = randint(0, 100)
            if val > 30:
                params = {'W': np.random.normal(size=(child.nn.layer_sizes[i], child.nn.layer_sizes[i - 1])),
                          'b': np.zeros((child.nn.layer_sizes[i], 1))}
                child.nn.change_layer_parameters(new_layer_parameters=params, layer_num=i)

    @staticmethod
    def get_probability_list(items):
        fitness = [item.fitness for item in items]
        total_fit = float(sum(fitness))
        relative_fitness = [f / total_fit for f in fitness]
        probabilities = [sum(relative_fitness[:i + 1])
                         for i in range(len(relative_fitness))]
        return probabilities
