#!/usr/bin/env python3

"""
DESCRIPTION
    SAmple implementation of genetic algorithms - for testing purposes 
LICENSE
    Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported 
    (CC BY-NC-SA 3.0)
    http://creativecommons.org/licenses/by-nc-sa/3.0/
AUTHOR
    Ales Daniel (ales.daniel@gmail.com)
"""


# code here
import random
import copy
import argparse
import sys

C_MIN = 97 #32  # ord() !
C_MAX = 125  # ord() }
F_VALUE = 1410  #2130  # 'Hello world' = 2133


class Param():
    """Class for using in argparse namespace
    """
    pass


class Crate():
    """A crate for individual and fitness value
    """
    def __init__(self, individual=None):
        self.individual = individual
        self.fitness_value = 0

    def set_fitness(self, fitness):
        self.fitness_value = fitness

    def set_individual(self, individual):
        self.individual = individual

    def get_individual(self):
        return self.individual

    def get_fitness(self):
        return self.fitness_value


class GenAlg():
    """ main class for genetic algorithm
    """
    
    # PUBLIC STATIC DATA MEMBERS
    cross_rate = 1.0
    mutation_rate = 0.05
    elitism_rate = 0.2
    # population_size = 10
    generations = 500
    individuals_count = 10
    population = []  # population of Crate objects

    def __init__(self, top_value, crossrate=None, mutationrate=None, 
            elitismrate=None, generations=None, individuals=None):
        if crossrate:
            self.cross_rate = float(crossrate)
        if mutationrate:
            self.mutation_rate = float(mutationrate)
        if elitismrate:
            self.elitism_rate = float(elitismrate)
        if generations:
            self.generations = int(generations)
        if individuals:
            self.individuals_count = int(individuals)
        self.peak = [top_value[i:i+1] for i in range(len(str(top_value)))]
    
    def init_population(self, first_population):
        if isinstance(first_population, (list, tuple)):
            if isinstance(first_population[0], Crate):
                self.population = first_population
        else:
            raise TypeError('First population is not list or tuple.')

    def add_individual(self, individual):
        """Adds new individual into population
        """
        if isinstance(individual, Crate):
            self.population.append(individual)
        else:
            c = Crate(individual)
            self.population.append(c)

    def _fitness(self, individual):
        """ fitness function for GA
        """
        peak = self.peak
        max_ = C_MAX - C_MIN + 1  # difference between possible chars
        value = 0  # fitness value
        for i in range(len(individual)):
            diff = abs(ord(individual[i]) - ord(peak[i]))
            value += max_ - diff
            if diff == 0:
                value += 100
        return value

    def _pick_individual(self, sum_fitness):
        """Selection in GA
        """
        r = random.randrange(0, sum_fitness+1)
        length = len(self.population)
        sum_fitness_ = 0
        #print('SumFitness', sum_fitness, 'r', r)
        #print('PopSize', len(self.population))
        for i in range(len(self.population)):
            #print(i, end=':')
            sum_fitness_ += self.population[i].get_fitness()
            if r <= sum_fitness_:
                #print('winner', i)
                winner = self.population[i]
                #print('>\t', self.print_individual(winner.get_individual()))
                return winner.get_individual()

    def _mutate(self, individual):
        """Mutation in GA
        """
        # no mutation
        if random.random() > self.mutation_rate:
            return individual
        # mutation
        r = random.random()
        if r <= 0.1:
            # mutate at 3 positions
            mut_ind = self._mutate_individual(individual, 3)
        elif r <= 0.4:
            # mutate at 2 positions
            mut_ind = self._mutate_individual(individual, 2)
        else:
            # mutate at 1 position
            mut_ind = self._mutate_individual(individual, 1)
        if len(individual) != len(self.peak):
            raise TypeError('Mutation mistake')
        return mut_ind

    def _mutate_individual(self, individual, positions_count=1):
        positions = []
        for i in range(positions_count):
            while True:
                p = random.randrange(0, len(individual))
                if not (p in positions):
                    break
            positions.append(p)
        #print('positions', positions)
        # mutation itself
        for i in positions:
            #individual[i] = chr(random.randrange(C_MIN, C_MAX+1))
            diff = random.randrange(-3, 3)
            if ord(individual[i]) + diff > C_MAX:
                # loop to start
                index = (ord(individual[i])+diff-C_MAX+C_MIN)
                if not (C_MIN <= index <= C_MAX):
                    raise TypeError('Index out of range in loop to start',
                            index)
                individual[i] = chr(index)
            elif ord(individual[i]) + diff < C_MIN:
                # loop to end
                index = (ord(individual[i])-C_MIN+diff+C_MAX)
                if not (C_MIN <= index <= C_MAX):
                    raise TypeError('Index out of range in loop to end', index)
                individual[i] = chr(index)
            else:
                index = (ord(individual[i])+diff)
                if not (C_MIN <= index <= C_MAX):
                    raise TypeError('Index out of range', index)
                individual[i] = chr(index)
            #print(individual)
        return individual

    def _crossover(self, individual1, individual2):
        """Crossover function in GA
        """
        if len(individual1) != len(individual2):
            raise TypeError("Individuals' length is not same.")

        # probability of crossing two individuals - no crossover
        if random.random() > self.cross_rate:
            return individual1, individual2
        r = random.random()
        if 0.2 >= r:
            child1, child2 = self._crossover_5050(individual1, individual2)
        elif r >= 0.8:
            child1, child2 = self._crossover_305020(individual1, individual2)
        else:
            child1, child2 = self._crossover_abab(individual1, individual2)
        if len(child1) != len(self.peak) or len(child2) != len(self.peak):
            raise TypeError('Crossover mistake.')
        return child1, child2

    def _crossover_5050(self, individual1, individual2):
        """Crossover by half
        """
        ind1_len = len(individual1)
        half = ind1_len // 2
        child1 = individual1[0:half]
        child1.extend(individual2[half:ind1_len])
        child2 = individual2[0:half]
        child2.extend(individual1[half:ind1_len])
        return child1, child2

    def _crossover_305020(self, individual1, individual2):
        """Crossover by 30:50:20 parts
        """
        ind1_len = len(individual1)
        part1 = int(ind1_len / 100 * 30)
        part3 = int(ind1_len / 100 * 20)
        part2 = ind1_len - part1 - part3
        child1 = individual1[0:part1] 
        child1.extend(individual2[part1:part1+part2])
        child1.extend(individual1[part1+part2:ind1_len])
        child2 = individual2[0:part1]
        child2.extend(individual1[part1:part1+part2])
        child2.extend(individual2[part1+part2:ind1_len])
        return child1, child2

    def _crossover_abab(self, individual1, individual2):
        """Crossover by shake - one two one two ...
        """
        child1 = []
        child2 = []
        for i in range(len(individual1)):
            child1.append(individual1[i])
            child2.append(individual2[i])
        return child1, child2

    def get_population(self):
        """Returns population
        """
        return self.population

    def print_individual(self, ind, end='\n'):
        output = ''
        for string in ind:
            output += string
        return output

    def print_individuals(self, text, ind1, ind2):
        print(text, end='  "')
        if ind1 != None:
            for i in ind1:
                print(i, end='')
        if ind2 != None:
            print('"     "', end='')
            for i in ind2:
                print(i, end='')
        print('"')

    def get_max_in_population(self):
        if self.population:
            return (sorted(self.population, key=lambda x: x.get_fitness(),
                    reverse=True))[0]

    def calc_fitness_in_population(self, sum_fitness=0):
        """Calculates fitness values for each individual in population
        """
        for ind in self.population:
            fit_value = self._fitness(ind.get_individual())
            ind.set_fitness(fit_value)
            sum_fitness += fit_value
        return sum_fitness

    def is_fitness_value_reached(self):
        if self.get_max_in_population().get_fitness() >= \
                self._fitness(self.peak):
            return True
        else:
            return False

    def breed_individuals(self, sum_fitness):
        """Selects individuals for crossover and mutate them and
        put them into new population
        """
        new_population = []

        # elitism part
        population_part_count = int(len(self.population) * self.elitism_rate)
        # move part to new pop and leave part divided by two
        if (self.individuals_count - population_part_count) % 2:
            population_part_count += 1
        # extend new population by first X percent of best individuals from
        # last population
        new_population.extend((sorted(self.population, key=lambda x:
            x.get_fitness(), reverse=True))[0:population_part_count])

        # classic select, crossover, mutate
        #self.population = self.population[population_part_count:]
        cycles = len(self.population) - population_part_count
        for i in range(cycles // 2):
            # select best-fit individuals
            #sum_fitness = self.calc_fitness_in_population(0)
            ind1 = self._pick_individual(sum_fitness)
            ind2 = self._pick_individual(sum_fitness)
            #   breed new individual fitness of new individuals
            child1, child2 = self._crossover(ind1, ind2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        if len(new_population) != self.individuals_count:
            raise TypeError('Count error in new population.')
        return new_population



if __name__ == '__main__':

    p = Param()
    parser = argparse.ArgumentParser(prog="hello",
                description='''
                Sample programme for genetic algorithm tests.
                Developed as a response to the initiative of Mira Hlavka and
                his programme in C :)
                ''',
                epilog='''
                This is Python version :)
                ''')
    parser.add_argument('-s', '--string', default='hello|world',
                        help="defines string to looking for (default "
                        "'hello|world')")
    parser.add_argument('-e', '--elitismrate', type=float, default=0.2,
                        help="defines elitism rate for picking best ones (default 0.2)",
                        metavar='FLOAT')
    parser.add_argument('-c', '--crossrate', type=float, default=1.0,
                        help="defines cross rate for crossover (default 1.0)",
                        metavar='FLOAT')
    parser.add_argument('-m', '--mutationrate', type=float, default=0.05,
                        help="defines mutation rate for mutation (default 0.05)",
                        metavar='FLOAT')
    parser.add_argument('-g', '--generations', type=int, default=500,
                        help="defines maximum generations (default 500)",
                        metavar='INTEGER')
    parser.add_argument('-r', '--random', type=int, default=0,
                        help="size of random individuals in first generation (default 0)",
                        metavar='INTEGER')
    parser.add_argument('-v', '--verbose', choices=('0', '1', '2'), default='0',
                        help="1 prints first generation,"
                        "\n2 prints population in every generation",
                        metavar='INTEGER')
    try:
        parser.parse_args(sys.argv[1:], namespace=p)
    except:
        sys.exit()
    p.verbose = int(p.verbose)

    ga = GenAlg(p.string, p.crossrate, p.mutationrate, p.elitismrate,
            p.generations, p.random)
    zeros = 0
    tmp = copy.copy(p.generations)
    while tmp // 10 > 0:
        zeros += 1
        tmp = tmp / 10
    # choose initial population
    if p.random == 0:
        initial_population_dumb = (
                '934 58v3875',
                'aw4n28a q22',
                ';.,49!o4@=k',
                ',./;[]{}!@#',
                'as%asd(f3(&',
                'm.*,g"3y1!4',
                'blekotybleb',
                ' ` ! @ # $ ',
                '-----------',
                'pa.wi,er}re')
        initial_population = []
        for string in initial_population_dumb:
            # change string to the list of strings
            string = [string[i:i+1] for i in range(len(string))]
            # modify string when it contains prohibited chars
            for i in range(len(string)):
                if not (C_MIN <= ord(string[i]) <= C_MAX):
                    string[i] = chr(random.randrange(C_MIN, C_MAX+1))
            c = Crate(string)
            initial_population.append(c)
        ga.init_population(initial_population)
    else:
        # random generation
        for i in range(p.random):
            ind = []
            for j in range(11):
                ind.append(chr(random.randrange(C_MIN, C_MAX+1))) 
            ga.add_individual(ind)

    # print first generation
    if p.verbose:
        print('First generation:')
        for ind in ga.get_population():
            print('\t', ga.print_individual(ind.get_individual()))

    # evaluate fitness of each individual in population
    last_gen = 0
    for i in range(ga.generations):
        # calc fitness for population and print best one
        sum_fitness = ga.calc_fitness_in_population()
        best =  ga.get_max_in_population()
        print('Best in {0} is {1} with fitness {2}'.format(
            i+1, ga.print_individual(best.get_individual()), best.get_fitness()))
        if p.verbose == 2:
            for ind in sorted(ga.get_population(), key=lambda x: x.get_fitness(),
                    reverse=True):
                print('\t', ga.print_individual(ind.get_individual()),
                        ind.get_fitness())
            input()


        # repeat on generation until termination (time limit, fitness value reached
        if ga.is_fitness_value_reached():
            last_gen = i + 1
            break

        # select, crossover, mutate
        new_population = ga.breed_individuals(sum_fitness)
       
        # replace least-fit population with new individuals
        ga.population = []
        for ind in new_population:
#            if len(ind) != len(ga.peak.get_individual()):
#                raise TypeError('Mistake in new population')
            ga.add_individual(ind)
    
    if not last_gen:
        last_gen = ga.generations

    # print first generation
    print('\nLast generation count:', last_gen)
    for ind in ga.get_population():
        ga.print_individual(ind.get_individual())

# vim:set sr et ts=4 sw=4 ft=python fenc=utf-8: // See Vim, :help 'modeline'
