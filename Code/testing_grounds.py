#Testing lambeq
##-----------------------------------------------------------------
#DisCoCat Syntax-based model
'''
from lambeq import BobcatParser
from discopy import grammar

sentence = 'John walks in the park'
# Parse the sentence and convert it into a string diagram
parser = BobcatParser(verbose='suppress')

diagram = parser.sentence2diagram(sentence)
grammar.draw(diagram, figsize=(14,3), fontsize=12) #Test Figure 1
'''
##-----------------------------------------------------------------
#Diagram rewriting
from lambeq import BobcatParser

# Parse the sentence
parser = BobcatParser(verbose='suppress')

diagram = parser.sentence2diagram("John walks in the park")
diagram.draw(figsize=(11,5), fontsize=13) #Test Figure 2
##-----------------------------------------------------------------
#Rewrite
from lambeq import Rewriter

rewriter = Rewriter(['prepositional_phrase', 'determiner'])
rewritten_diagram = rewriter(diagram)

#eliminates "the" -connectors- from the diagram
rewritten_diagram.draw(figsize=(11,5), fontsize=13) #Test Figure 3

