# %%
import numpy as np
from ik_problem import IKProblem


class Individual:
    def __init__(self, bounds):
        self.genes = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(bounds.shape[0],)
        )
        self.gradients = np.zeros(bounds.shape[0])
        self.extinction = 0
        self.fitness = 0


class MemOpt:
    def __init__(
        self,
        bounds,
        initial_guess,
        n_eval=50,
        population_size=10,
        elite_size=4,
        problem=None,
        params=None,
    ):
        self.bounds = bounds
        self.n_genes = bounds.shape[0]
        self.n_eval = n_eval
        self.population_size = population_size
        self.elite_size = elite_size
        self.population = []

        self.best_fitness_history = []
        self.best_fitness_change_history = []

        self.initial_guess = initial_guess
        self.solution = initial_guess
        self.best_fitness = 0
        self.best_genes = np.zeros(self.n_genes)
        self.best_gradients = np.zeros(self.n_genes)
        self.compute_fitness = problem.fitness
        self.compute_grad = problem.gradient
        self.problem = problem
        self.solution_found = False
        self.eps_p = params["eps.pos"]
        self.eps_o = params["eps.ori"]
        self.eps_r = params["eps.rcm"]
        self.print_level = params["print_level"]
        self.mode = params["mode"]

    def compute_fitness(self, genes):
        raise NotImplementedError

    def set_verbosity(self, level):
        self.print_level = level

    def update_extinction(self):
        min_fitness = self.population[0].fitness
        max_fitness = self.population[-1].fitness

        for i in range(self.population_size):
            self.population[i].extinction = (
                self.population[i].fitness
                + min_fitness * ((i / (self.population_size - 1)) - 1)
            ) / max_fitness

    def print_gene_vector(self, genes):
        genes_str = "["
        for j in range(len(genes)):
            genes_str += "{:.3f}".format(genes[j])
            if j < len(genes) - 1:
                genes_str += ", "
        genes_str += "]"
        return genes_str

    def clip_gene(self, gene, i):
        if gene < self.bounds[i, 0]:
            gene = self.bounds[i, 0]
        elif gene > self.bounds[i, 1]:
            gene = self.bounds[i, 1]
        return gene

    def exploit(self, individual):
        grads = self.compute_grad(individual.genes)
        delta = 1.0

        genes_base = individual.genes.copy()
        fitness_base = individual.fitness

        genes_var = genes_base - delta * grads
        for gene in range(self.n_genes):
            self.clip_gene(genes_var[gene], gene)

        individual.genes = genes_var.copy()
        individual.fitness = self.compute_fitness(individual.genes)[0]

        if individual.fitness > fitness_base:
            individual.genes = genes_base.copy()
            individual.fitness = fitness_base
        else:
            individual.gradients = (
                np.random.uniform() * individual.gradients.copy()
                + np.random.uniform() * (genes_var - genes_base)
            )

    def gene_crossover(self, parent_gene_1, parent_gene_2, crossover_ratio):
        return crossover_ratio * parent_gene_1 + (1 - crossover_ratio) * parent_gene_2

    def compute_mutation_ratio(self, parent1, parent2):
        average_extinction = (parent1.extinction + parent2.extinction) / 2
        return average_extinction * (1 - 1 / self.n_genes) + 1 / self.n_genes

    def reproduce(self, offspring, parent1, parent2, prototype):
        # Compute mutation ratio
        mutation_ratio = self.compute_mutation_ratio(parent1, parent2)
        average_extinction = (parent1.extinction + parent2.extinction) / 2

        for gene in range(self.n_genes):
            # Crossover
            crossover_ratio = np.random.uniform()
            offspring.genes[gene] = self.gene_crossover(
                parent1.genes[gene], parent2.genes[gene], crossover_ratio
            )

            offspring.genes[gene] += parent1.gradients[gene] * np.random.uniform()
            offspring.genes[gene] += parent2.gradients[gene] * np.random.uniform()

            store_gene = offspring.genes[gene]

            # Mutation
            if np.random.uniform() < mutation_ratio:
                offspring.genes[gene] += (
                    np.random.uniform(-1, 1)
                    * average_extinction
                    * (self.bounds[gene, 1] - self.bounds[gene, 0])
                )

            # Adoption
            average_genes = (parent1.genes[gene] + parent2.genes[gene]) / 2
            offspring.genes[gene] += self.gene_crossover(
                np.random.uniform() * (average_genes - offspring.genes[gene]),
                np.random.uniform() * (prototype.genes[gene] - offspring.genes[gene]),
                np.random.uniform(),
            )

            # Clip gene
            offspring.genes[gene] = self.clip_gene(offspring.genes[gene], gene)

            # Update gradients
            offspring.gradients[gene] = np.random.uniform() * offspring.gradients[
                gene
            ] + (offspring.genes[gene] - store_gene)

        offspring.fitness = self.compute_fitness(offspring.genes)[0]

    def linear_dist(self, n):
        if n <= 0:
            return 0
        dist = np.random.randint(0, n * (n + 3) // 2)

        if dist == 0:
            return 0
        i = -1
        while dist > 0:
            dist -= n - i
            i += 1
        return i

    def evolution_step(self, population):
        # Assign whole population to the mating pool
        mating_pool = self.population.copy()
        offspring = self.population.copy()

        # Perfom explotation for elite infividuals
        for i in range(self.elite_size):
            self.exploit(offspring[i])

            if self.print_level > 2:
                print("++++++++ Exploitation of elite individuals")
                # Print fitness values and genes of offspring individuals
                genes_str = "["
                for j in range(len(offspring[i].genes)):
                    genes_str += "{:.3f}".format(offspring[i].genes[j])
                    if j < len(offspring[i].genes) - 1:
                        genes_str += ", "
                genes_str += "]"
                print(
                    "Offspring {} fitness: {:.3f} genes: {}".format(
                        i, offspring[i].fitness, genes_str
                    )
                )

        for i in range(self.elite_size, self.population_size):
            if self.print_level > 2:
                print("++++++++ Reproduction of offspring")
                print("mating pool size: ", len(mating_pool))

            # Check if mating pool is empty
            if len(mating_pool) != 0:
                # Selecting parents IDs from a linear integer ditribution
                parent1_id = self.linear_dist(len(mating_pool) - 1)
                parent2_id = self.linear_dist(len(mating_pool) - 1)
                prototype_id = self.linear_dist(len(mating_pool) - 1)

                # Select the random individuals from the mating pool
                parent1 = mating_pool[parent1_id]
                parent2 = mating_pool[parent2_id]
                prototype = mating_pool[prototype_id]

                # Print fitness values and genes of parents
                if self.print_level > 2:
                    print(
                        "parent1_id: ",
                        parent1_id,
                        "\tparent2_id: ",
                        parent2_id,
                        "\tprototype_id: ",
                        prototype_id,
                    )

                    genes_str = "["
                    for j in range(len(parent1.genes)):
                        genes_str += "{:.3f}".format(parent1.genes[j])
                        if j < len(parent1.genes) - 1:
                            genes_str += ", "
                    genes_str += "]"
                    print(
                        "parent1 fitness: {:.3f} genes: {}".format(
                            parent1.fitness, genes_str
                        )
                    )

                    genes_str = "["
                    for j in range(len(parent2.genes)):
                        genes_str += "{:.3f}".format(parent2.genes[j])
                        if j < len(parent2.genes) - 1:
                            genes_str += ", "
                    genes_str += "]"
                    print(
                        "parent2 fitness: {:.3f} genes: {}".format(
                            parent2.fitness, genes_str
                        )
                    )

                    genes_str = "["
                    for j in range(len(prototype.genes)):
                        genes_str += "{:.3f}".format(prototype.genes[j])
                        if j < len(prototype.genes) - 1:
                            genes_str += ", "
                    genes_str += "]"
                    print(
                        "prototype fitness: {:.3f} genes: {}".format(
                            prototype.fitness, genes_str
                        )
                    )

                # Reproduce
                self.reproduce(offspring[i], parent1, parent2, prototype)

                # Print fitness values and genes of offspring individual
                if self.print_level > 2:
                    print("++++++ Reproducing individual ", i)
                    genes_str = "["
                    for j in range(len(offspring[i].genes)):
                        genes_str += "{:.3f}".format(offspring[i].genes[j])
                        if j < len(offspring[i].genes) - 1:
                            genes_str += ", "
                    genes_str += "]"
                    print(
                        "Offspring fitness: {:.3f} genes: {}".format(
                            offspring[i].fitness, genes_str
                        )
                    )

                # Remove parents from mating pool with higher fitness than offspring
                if offspring[i].fitness < parent1.fitness:
                    mating_pool.remove(parent1)
                if offspring[i].fitness < parent2.fitness and parent1 != parent2:
                    mating_pool.remove(parent2)
            else:
                print("++++++ mating pool empty. creating new individual ", i)
                offspring[i] = Individual(self.bounds)
                offspring[i].fitness = self.compute_fitness(offspring[i].genes)[0]

        self.population = offspring.copy()

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)

        # Print fitness values and genes of offspring individuals
        if self.print_level > 1:
            print("---------------New population------------")

        # Print fitness values and genes of all offspring individuals with 3 decimals
        for i in range(self.population_size):
            genes_str = self.print_gene_vector(self.population[i].genes)
            gradients_str = self.print_gene_vector(self.population[i].gradients)

            population.set_xf(i, self.population[i].genes, [self.population[i].fitness])

            if i == 0:
                sol_found, err_p, err_o, err_r = self.check_convergence(
                    self.population[i].genes
                )

            if self.print_level > 1:
                print(
                    "[{}] fitness: {:.6f} \tperr: {:.5f} \toerr: {:.5f} \trerr: {:.5f} \tgenes: {} \tgradients: {}".format(
                        i,
                        self.population[i].fitness,
                        err_p,
                        err_o,
                        err_r,
                        genes_str.ljust(30),
                        gradients_str.ljust(100),
                    )
                )

            if sol_found:
                population.set_xf(
                    0, self.population[i].genes, [self.population[i].fitness]
                )

                if self.print_level > 0:
                    print(
                        f"Solution found in generation {str(self.gen_id)} fitness:{self.population[i].fitness} perro: {err_p} oerr: {err_o} rerr: {err_r}"
                    )
                self.solution_found = True
                self.best_genes = self.population[i].genes
                return True

        #  Update extinction
        self.update_extinction()

        # Update solution
        self.update_solution()

        return False

    def update_solution(self):
        if self.population[0].fitness < self.best_fitness:
            self.best_genes = self.population[0].genes
            self.best_fitness = self.population[0].fitness
            genes_str = self.print_gene_vector(self.best_genes)
            if self.print_level > 0:
                print("Updated solution: ", genes_str)

    def wipe_out(self):
        # Wipe out all individuals except the best one
        self.population = [Individual(self.bounds) for _ in range(self.population_size)]
        self.population[0].genes = self.best_genes
        self.population[0].fitness = self.best_fitness

        # Update fitness of all individuals
        for i in range(self.population_size):
            self.population[i].fitness = self.compute_fitness(self.population[i].genes)[
                0
            ]

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)

        # Print fitness values and genes of all individuals with 3 decimals
        for i in range(self.population_size):
            genes_str = self.print_gene_vector(self.population[i].genes)
            gradients_str = self.print_gene_vector(self.population[i].gradients)

            _, err_p, err_o, err_r = self.check_convergence(self.population[i].genes)

            if self.print_level > 1:
                print(
                    "[{}] fitness: {:.6f} \tperr: {:.5f} \toerr: {:.5f} \trerr: {:.5f} \tgenes: {} \tgradients: {}".format(
                        i,
                        self.population[i].fitness,
                        err_p,
                        err_o,
                        err_r,
                        genes_str.ljust(30),
                        gradients_str.ljust(100),
                    )
                )
        self.update_extinction()
        self.update_solution()

    # Verify if convergence criteria is met
    def check_convergence(self, individual_genes):
        prob = self.problem.extract(IKProblem)
        prob.update_model(individual_genes)

        err_p = prob.perr()
        err_o = prob.oerr()
        _, _, _, _, err_r = prob.compute_residual_RCM()

        if (
            self.mode in ["c6d"]
            and err_p < self.eps_p
            and err_o < self.eps_o
            and err_r < self.eps_r
        ):
            if self.print_level > 0:
                print("Convergence criteria met!")
            self.solution_found = True
        elif self.mode in ["u3d"] and err_p < self.eps_p:
            if self.print_level > 0:
                print("Convergence criteria met!")
            self.solution_found = True
        elif self.mode in ["u6d"] and err_p < self.eps_p and err_o < self.eps_o:
            if self.print_level > 0:
                print("Convergence criteria met!")
            self.solution_found = True

        else:
            self.solution_found = False

        return self.solution_found, err_p, err_o, err_r

    def evolve(self, population):
        individuals_matrix = population.get_x()
        self.best_genes = self.initial_guess.copy()
        self.best_fitness = self.compute_fitness(self.initial_guess)[0]
        if self.print_level > 0:
            print(f"Initial guess: {self.initial_guess}")
            print(f"Initial fitness: {self.best_fitness}")

        # Initialize population
        self.population = [Individual(self.bounds) for _ in range(self.population_size)]

        for id, indiv in enumerate(self.population):
            # self.population[id].genes = self.initial_guess[id].copy()
            self.population[id].genes = individuals_matrix[id].copy()

        # Update fitness of all individuals
        for i in range(self.population_size):
            self.population[i].fitness = self.compute_fitness(self.population[i].genes)[
                0
            ]

        # Add initial guess to population
        self.population[0].genes = self.solution
        self.population[0].fitness = self.compute_fitness(self.population[0].genes)[0]

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)

        # Print all fitness values and genes
        for i in range(self.population_size):
            genes_str = self.print_gene_vector(self.population[i].genes)
            gradients_str = self.print_gene_vector(self.population[i].gradients)

            sol_found, err_p, err_o, err_r = self.check_convergence(
                self.population[i].genes
            )

            if self.print_level > 1:
                print(
                    "[{}] fitness: {:.6f} ".format(
                        i,
                        self.population[i].fitness,
                        err_p,
                        err_o,
                        err_r,
                        genes_str.ljust(30),
                        gradients_str.ljust(100),
                    )
                )

            if sol_found:
                if self.print_level > 1:
                    print(
                        f"Solution found before evolution {str(self.gen_id)} perro: {err_p} oerr: {err_o} rerr: {err_r}"
                    )
                self.solution_found = True
                self.best_genes = self.population[i].genes
                return True

        self.update_extinction()

        fitness_change_threshold = 0.00001
        fitness_nochange_counter = 0
        finess_nochange_limit = 10

        res = False

        # Evolution loop
        for i in range(self.n_eval):
            if self.print_level > 0:
                print("\n\nGeneration: ", i)
            self.gen_id = i

            res = self.evolution_step(population=population)
            if res:
                break

            # Store best fitness and errors
            self.best_fitness_history.append(self.best_fitness)
            if self.gen_id == 0:
                self.best_fitness_change_history.append(0)
            else:
                self.best_fitness_change_history.append(
                    self.best_fitness_history[-1] - self.best_fitness_history[-2]
                )

            # Check if fitness is not changing
            if self.gen_id > 0:
                if (
                    np.abs(self.best_fitness_change_history[-1])
                    < fitness_change_threshold
                ):
                    fitness_nochange_counter += 1
                else:
                    fitness_nochange_counter = 0
            if self.print_level > 0:
                print(
                    "Best fitness change: ",
                    np.abs(self.best_fitness_change_history[-1]),
                )

            # If fitness is not changing for a while, wipe out the population
            if fitness_nochange_counter > finess_nochange_limit:
                if self.print_level > 0:
                    print("Wiping out the population in generation ", i)
                self.wipe_out()
                fitness_nochange_counter = 0

            # Print best fitness and genes
            genes_str = self.print_gene_vector(self.best_genes)

            if self.print_level > 0:
                print(
                    "Best fitness until Gen ",
                    i,
                    ": ",
                    self.best_fitness,
                    "Best genes: ",
                    genes_str,
                )
        return population
