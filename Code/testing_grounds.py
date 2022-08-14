#Testing lambeq
##-----------------------------------------------------------------
#DisCoCat Syntax-based model
from lambeq import BobcatParser
from discopy import grammar

sentence = 'John walks in the park'
# Parse the sentence and convert it into a string diagram
parser = BobcatParser(verbose='suppress')

diagram = parser.sentence2diagram(sentence)
#grammar.draw(diagram, figsize=(14,3), fontsize=12) #Test Figure 1

##-----------------------------------------------------------------
#Diagram rewriting
from lambeq import BobcatParser

# Parse the sentence
parser = BobcatParser(verbose='suppress')

diagram = parser.sentence2diagram("John walks in the park")
#diagram.draw(figsize=(11,5), fontsize=13) #Test Figure 2
##-----------------------------------------------------------------
#Rewrite
from lambeq import Rewriter

rewriter = Rewriter(['prepositional_phrase', 'determiner'])
rewritten_diagram = rewriter(diagram)

#eliminates "the" -connectors- from the diagram
#rewritten_diagram.draw(figsize=(11,5), fontsize=13) #Test Figure 3
##-----------------------------------------------------------------
#Normalize diagram
normalised_diagram = rewritten_diagram.normal_form()
#normalised_diagram.draw(figsize=(9,4), fontsize=13) #Test Figure 4
##-----------------------------------------------------------------
#Curry Rewrite Rule: input wires up, output wires down - distinction
curry_functor = Rewriter(['curry'])

curried_diagram = curry_functor(normalised_diagram)
#curried_diagram.draw(figsize=(9,4), fontsize=13) #Test Figure 5

#normalization curry results -> faster execution
#curried_diagram.normal_form().draw(figsize=(5,4), fontsize=13) #Test Figure 6
##-----------------------------------------------------------------
from lambeq import BobcatParser

sentence = 'John walks in the park'

# Get a string diagram
parser = BobcatParser(verbose='text')
diagram = parser.sentence2diagram(sentence) 
#diagram.draw(figsize=(9,4), fontsize=13) #Test Figure 7
#looks the same as Test Figure 2
##-----------------------------------------------------------------
from lambeq import AtomicType, IQPAnsatz

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Convert string diagram to quantum circuit
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=2)
discopy_circuit = ansatz(diagram)
#discopy_circuit.draw(figsize=(15,10)) #Test Figure 8

