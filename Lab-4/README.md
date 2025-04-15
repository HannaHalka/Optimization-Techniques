# Knapsack-problem

***

`generate_population: ` Generate matrix $\text{population_size} \times \text{chromosome_length}$.

`fitness: ` We calculate weights and benefits for all chromosome.

`selection: ` Looking at fitness_scores we choose six best chromosome and six worst. 

`crossover_and_mutation: ` Six best chromosome makes 2 children. (using middle point) 

`replace: ` Six worst chromosome replacing to six new children.

`evolution: `

***

$\text{weights}^T$ = 
[4  <span style="color: red;"> 3 6 2 </span> 1  8  5  2 <span style="color: red;"> 7  6  3 </span>  4  2  1  9  3  2  <span style="color: red;"> 1 </span>  4  <span style="color: red;"> 6 </span>]

$\text{benefits}^T$ = 
[4  <span style="color: red;"> 5 6 7 </span> 0  0  0  0 <span style="color: red;"> 7  6  5 </span> 4  0  0  0  0  3  <span style="color: red;"> 9 </span> 2  <span style="color: red;"> 8 </span>]

Tacking into a count weights vector and benefits vector:
`Weight ` = 34,
`The best fitness function ` = 53

***

Best fitness = 40

Chromosome$^T$= [0 1 1 1 0 0 1 1 1 1 0 0 1 1 0 0 0 1 0 0]
