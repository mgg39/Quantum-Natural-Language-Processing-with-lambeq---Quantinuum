#Testing lambeq
##-----------------------------------------------------------------
#DisCoCat Syntax-based model
from lambeq import BobcatParser
from discopy import grammar

sentence = 'John walks in the park'

# Parse the sentence and convert it into a string diagram
parser = BobcatParser(verbose='suppress')
diagram = parser.sentence2diagram(sentence)

grammar.draw(diagram, figsize=(14,3), fontsize=12) #Test Figure 1

##-----------------------------------------------------------------